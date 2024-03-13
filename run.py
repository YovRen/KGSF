import json

from tqdm import tqdm
import pickle as pkl
import torch
from crsdataset import CRSDataset
from crsmodel import CrossModel
from nltk.translate.bleu_score import sentence_bleu
import warnings

warnings.filterwarnings("ignore")


class TrainLoop:
    def __init__(self):
        self.crs_data_path = "data"
        self.batch_size = 128
        self.learningrate = 0.001
        self.gradient_clip = 0.1
        self.optimizer = 'adam'
        self.device = 'cuda'
        self.n_user = 1075
        self.n_concept = 29308
        self.n_entity = 64368
        self.n_relation = 214  # 46+18
        self.n_con_relation = 48
        self.dim = 128
        self.n_hop = 2
        self.n_bases = 8
        self.hidden_dim = 128
        self.max_c_length = 256
        self.max_r_length = 30
        self.embedding_size = 300
        self.n_heads = 2
        self.n_layers = 2
        self.ffn_size = 300
        self.n_relation = 64
        self.n_user = 1075
        self.dropout = 0.1
        self.attention_dropout = 0.0
        self.relu_dropout = 0.1
        self.entity2entityId = pkl.load(open('data/entity2entityId.pkl', 'rb'))
        self.id2entity = pkl.load(open('data/id2entity.pkl', 'rb'))
        self.text_dict = pkl.load(open('data/text_dict.pkl', 'rb'))
        self.userId2userIdx = json.load(open(self.crs_data_path + '/redial_userId2userIdx.jsonl', encoding='utf-8'))
        self.dbpedia_subkg = json.load(open(self.crs_data_path + '/dbpedia_subkg.jsonl', encoding='utf-8'))
        # self.train_dataset=CRSDataset('toy_train', self)
        # self.test_dataset = CRSDataset('toy_test', self)
        # self.valid_dataset = CRSDataset('toy_valid', self)
        self.train_dataset = CRSDataset('train', self)
        self.test_dataset = CRSDataset('test', self)
        self.valid_dataset = CRSDataset('valid', self)
        self.train_dataloader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        self.valid_dataloader = torch.utils.data.DataLoader(dataset=self.valid_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        self.metrics_rec = {"rec_loss": 0, "recall@1": 0, "recall@10": 0, "recall@50": 0, "count": 0}
        self.metrics_gen = {"gen_loss": 0, "dist1": 0, "dist2": 0, "dist3": 0, "dist4": 0, "bleu1": 0, "bleu2": 0, "bleu3": 0, "bleu4": 0, "count": 0}
        self.dict = self.train_dataset.word2index
        self.movie_ids = pkl.load(open("data/movie_ids.pkl", "rb"))
        self.index2word = {self.dict[key]: key for key in self.dict}
        self.model = CrossModel(self, self.dict).to(self.device)
        self.optimizer = {k.lower(): v for k, v in torch.optim.__dict__.items() if not k.startswith('__') and k[0].isupper()}[self.optimizer]([p for p in self.model.parameters() if p.requires_grad], lr=self.learningrate, amsgrad=True, betas=(0.9, 0.999))

    def train(self, rec_epoch, gen_epoch):
        losses = []
        best_val = 10000
        for i in range(rec_epoch + gen_epoch):
            self.model.train()
            for num, (context, response, entity, dbpedia_mentioned, user_mentioned, movie, concept_mask, dbpedia_mask, concept_vector, dbpedia_vector, user_vector, rec) in enumerate(tqdm(self.train_dataloader)):
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                self.optimizer.zero_grad()
                _, _, _, rec_loss, gen_loss, mi_loss = self.model(context.to(self.device), response.to(self.device), concept_mask, dbpedia_mask, seed_sets, movie, concept_vector.to(self.device), dbpedia_vector.to(self.device), user_vector.to(self.device), dbpedia_mentioned.to(self.device), user_mentioned.to(self.device), rec)
                if i < rec_epoch:
                    joint_loss = rec_loss
                else:
                    joint_loss = gen_loss
                joint_loss.backward()

                self.optimizer.step()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                losses.append([gen_loss, rec_loss, mi_loss, joint_loss])
                if (num + 1) % (512 / self.batch_size) == 0:
                    print('gen_loss is %f' % (sum([l[0] for l in losses]) / len(losses)))
                    print('rec_loss is %f' % (sum([l[1] for l in losses]) / len(losses)))
                    print('mi_loss is %f' % (sum([l[2] for l in losses]) / len(losses)))
                    print('joint_loss is %f' % (sum([l[3] for l in losses]) / len(losses)))
                    losses = []
            output_metrics_rec, output_metrics_gen = self.val()
            if i < rec_epoch:
                if best_val < output_metrics_rec["rec_loss"]:
                    break
                else:
                    best_val = output_metrics_rec["rec_loss"]
                    self.model.save_model('rec')
                    print("recommendation model saved once------------------------------------------------")
            elif i == rec_epoch:
                best_val = output_metrics_gen["gen_loss"]
                self.model.save_model('gen')
                print("generator model saved once------------------------------------------------")
            else:
                if best_val < output_metrics_gen["gen_loss"]:
                    break
                else:
                    best_val = output_metrics_gen["gen_loss"]
                    self.model.save_model('gen')
                    print("generator model saved once------------------------------------------------")

    def val(self, is_test=False):
        self.model.eval()
        val_dataloader = self.test_dataloader if is_test else self.valid_dataloader

        def vector2sentence(batch_sen):
            sentences = []
            for sen in batch_sen.numpy().tolist():
                sentence = []
                for word in sen:
                    if word > 3:
                        sentence.append(self.index2word[word])
                    elif word == 3:
                        sentence.append('_UNK_')
                sentences.append(sentence)
            return sentences

        tokens_response = []
        tokens_predict = []
        tokens_context = []
        for context, response, entity, dbpedia_mentioned, user_mentioned, movie, concept_mask, dbpedia_mask, concept_vector, dbpedia_vector, user_vector, rec in tqdm(val_dataloader):
            with torch.no_grad():
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                _, _, rec_scores, rec_loss, gen_loss, _ = self.model(context.to(self.device), response.to(self.device), concept_mask, dbpedia_mask, seed_sets, movie, concept_vector.to(self.device), dbpedia_vector.to(self.device), user_vector.to(self.device), dbpedia_mentioned.to(self.device), user_mentioned.to(self.device), rec)
                scores, preds, _, _, _, _, _ = self.model(context.to(self.device), None, concept_mask, dbpedia_mask, seed_sets, movie, concept_vector.to(self.device), dbpedia_vector.to(self.device), user_vector.to(self.device), dbpedia_mentioned.to(self.device), user_mentioned.to(self.device), rec)
            tokens_response.extend(vector2sentence(response.cpu()))
            tokens_predict.extend(vector2sentence(preds.cpu()))
            tokens_context.extend(vector2sentence(context.cpu()))
            self.metrics_gen['gen_loss'] += gen_loss
            self.metrics_rec["rec_loss"] += rec_loss
            _, pred_idx = torch.topk(rec_scores.cpu()[:, torch.LongTensor(self.movie_ids)], k=100, dim=1)
            for b in range(context.shape[0]):
                if movie[b].item() == 0:
                    continue
                target_idx = self.movie_ids.index(movie[b].item())
                self.metrics_rec["recall@1"] += int(target_idx in pred_idx[b][:1].tolist())
                self.metrics_rec["recall@10"] += int(target_idx in pred_idx[b][:10].tolist())
                self.metrics_rec["recall@50"] += int(target_idx in pred_idx[b][:50].tolist())
                self.metrics_rec["count"] += 1
        for out, tar in zip(tokens_predict, tokens_response):
            self.metrics_gen['bleu1'] += sentence_bleu([tar], out, weights=(1, 0, 0, 0))
            self.metrics_gen['bleu2'] += sentence_bleu([tar], out, weights=(0, 1, 0, 0))
            self.metrics_gen['bleu3'] += sentence_bleu([tar], out, weights=(0, 0, 1, 0))
            self.metrics_gen['bleu4'] += sentence_bleu([tar], out, weights=(0, 0, 0, 1))
        unigram_count = 0
        bigram_count = 0
        trigram_count = 0
        quagram_count = 0
        unigram_set = set()
        bigram_set = set()
        trigram_set = set()
        quagram_set = set()
        # outputs is a list which contains several sentences, each sentence contains several words
        for sen in tokens_predict:
            for word in sen:
                unigram_count += 1
                unigram_set.add(word)
            for start in range(len(sen) - 1):
                bg = str(sen[start]) + ' ' + str(sen[start + 1])
                bigram_count += 1
                bigram_set.add(bg)
            for start in range(len(sen) - 2):
                trg = str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2])
                trigram_count += 1
                trigram_set.add(trg)
            for start in range(len(sen) - 3):
                quag = str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2]) + ' ' + str(sen[start + 3])
                quagram_count += 1
                quagram_set.add(quag)
        self.metrics_gen['dist1'] = len(unigram_set) / len(tokens_predict)  # unigram_count
        self.metrics_gen['dist2'] = len(bigram_set) / len(tokens_predict)  # bigram_count
        self.metrics_gen['dist3'] = len(trigram_set) / len(tokens_predict)  # trigram_count
        self.metrics_gen['dist4'] = len(quagram_set) / len(tokens_predict)  # quagram_count
        text_response = [' '.join(tokens) for tokens in tokens_response]
        text_predict = [' '.join(tokens) for tokens in tokens_predict]
        text_context = [' '.join(tokens) for tokens in tokens_context]
        with open('test_gen.txt', 'w', encoding='utf-8') as file:
            for context, predict, response in zip(text_context, text_predict, text_response):
                file.writelines('=' * 100 + '\n')
                file.writelines("context:" + context + '\n')
                file.writelines("response:" + response + '\n')
                file.writelines("predict:" + predict + '\n')
        self.metrics_rec = {key: self.metrics_rec[key] / self.metrics_rec['count'] for key in self.metrics_rec}
        self.metrics_gen = {key: self.metrics_gen[key] / (self.batch_size * len(val_dataloader)) if 'bleu' in key or 'loss' in key else self.metrics_gen[key] for key in self.metrics_gen}
        print(self.metrics_rec)
        print(self.metrics_gen)
        return self.metrics_rec, self.metrics_gen


if __name__ == '__main__':
    loop = TrainLoop()
    loop.model.load_model('rec')
    loop.train(rec_epoch=1, gen_epoch=1)
    met = loop.val(is_test=True)
