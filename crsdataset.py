from collections import defaultdict

from torch.utils.data.dataset import Dataset
import numpy as np
from tqdm import tqdm
import json
import nltk
from copy import deepcopy
import gensim
import torch
import pickle as pkl


# 1.

class CRSDataset(Dataset):
    def __init__(self, mode, args):

        self.crs_data_path = args.crs_data_path
        self.batch_size = args.batch_size
        self.max_c_length = args.max_c_length
        self.max_r_length = args.max_r_length
        self.n_entity = args.n_entity
        self.n_concept = args.n_concept + 1
        self.n_user = args.n_user
        self.entity2entityId = args.entity2entityId
        self.userId2userIdx = args.userId2userIdx
        self.special_wordIdx = args.special_wordIdx
        self.id2entity = args.id2entity
        self.text_dict = args.text_dict
        self.entity_max = len(self.entity2entityId)
        self.concept_min = 0
        # self.prepare_dbpedia_subkg()
        f = open(self.crs_data_path + '/' + mode + '_data.jsonl', encoding='utf-8')
        self.cases = []
        for case in tqdm(f):
            lines = json.loads(case.strip())
            initiatorWorkerId = lines["initiatorWorkerId"]
            respondentWorkerId = lines["respondentWorkerId"]
            messages = lines['messages']
            movieMentions = lines['movieMentions']
            respondentQuestions = lines['respondentQuestions']
            initiatorQuestions = lines['initiatorQuestions']
            cases = self._context_reformulate(messages, movieMentions, respondentQuestions, initiatorQuestions, initiatorWorkerId, respondentWorkerId)
            self.cases.extend(cases)

        self.word2index = json.load(open('data/word2index_redial.json', encoding='utf-8'))
        self.key2index = json.load(open('data/key2index_3rd.json', encoding='utf-8'))
        self.stopwords = set([word.strip() for word in open('data/stopwords.txt', encoding='utf-8')])
        # self.prepare_word2vec()
        self.datapre = self.data_process(is_finetune=False)
        # self.co_occurance_ext(self.cases)  # exit()
        # self.datapre = self.datapre[:len(self.datapre) // 5]

    def _context_reformulate(self, messages, movieMentions, respondentQuestions, initiatorQuestions, initiatorWorkerId, respondentWorkerId):
        """
        对数据进行重组和整理，以生成训练样本
        """
        lastId = None
        message_list = []
        for message in messages:
            entities = []
            try:
                for entity in self.text_dict[message['text']]:
                    entities.append(self.entity2entityId[entity])
            except:
                pass
            # 用于检测消息中提及的电影并转换成特定格式
            token_text_ori = nltk.word_tokenize(message['text'])
            token_text = []
            num = 0
            # 对分词结果中的 @ abc 合成一个单词，得到 token_text
            while num < len(token_text_ori):
                if token_text_ori[num] == '@' and num + 1 < len(token_text_ori):
                    token_text.append(token_text_ori[num] + token_text_ori[num + 1])
                    num += 2
                else:
                    token_text.append(token_text_ori[num])
                    num += 1
            movie_rec_ori = []
            for word in token_text:
                if word[1:] in movieMentions:
                    movie_rec_ori.append(word[1:])
            movie_rec = []
            # 将句子中出现的 id 转化为 entityId，得到 movie_rec
            for movie in movie_rec_ori:
                try:
                    entity = self.id2entity[int(movie)]
                    movie_rec.append(self.entity2entityId[entity])
                except:
                    pass
            if len(message_list) == 0:
                message_dict = {'text': token_text, 'entity': entities + movie_rec, 'user': message['senderWorkerId'], 'movie': movie_rec}
                message_list.append(message_dict)
                lastId = message['senderWorkerId']
                continue
            if message['senderWorkerId'] == lastId:
                message_list[-1]['text'] += token_text
                message_list[-1]['entity'] += entities + movie_rec
                message_list[-1]['movie'] += movie_rec
            else:
                message_dict = {'text': token_text, 'entity': entities + movie_rec, 'user': message['senderWorkerId'], 'movie': movie_rec}
                message_list.append(message_dict)
                lastId = message['senderWorkerId']
        cases = []
        contexts = []
        entities = []
        users = []
        for message_dict in message_list:
            if message_dict['user'] == respondentWorkerId and len(contexts) > 0:
                response = message_dict['text']
                if len(message_dict['movie']) != 0:
                    for movie in message_dict['movie']:
                        cases.append({'contexts': deepcopy(contexts), 'response': response, 'users': users, 'user': message_dict['user'], 'entity': deepcopy(entities), 'movie': movie, 'rec': 1})
                else:
                    cases.append({'contexts': deepcopy(contexts), 'response': response, 'users': users, 'user': message_dict['user'], 'entity': deepcopy(entities), 'movie': 0, 'rec': 0})
            contexts.append(message_dict['text'])
            users.append(message_dict['user'])
            for word in message_dict['entity']:
                if word not in entities:
                    entities.append(word)
        return cases

    def prepare_word2vec(self):
        """
        准备 Word2Vec 模型，对语料库进行处理并生成词向量
        """
        corpus = []
        for mode in ['train', 'test', 'valid']:
            f = open(self.crs_data_path + '/' + mode + '_data.jsonl', encoding='utf-8')
            for case in tqdm(f):
                lines = json.loads(case.strip())
                messages = lines['messages']
                for message in messages:
                    token_text_ori = nltk.word_tokenize(message['text'])
                    token_text = []
                    num = 0
                    # 对分词结果中的 @ abc 合成一个单词，得到 token_text
                    while num < len(token_text_ori):
                        if token_text_ori[num] == '@' and num + 1 < len(token_text_ori):
                            token_text.append(token_text_ori[num] + token_text_ori[num + 1])
                            num += 2
                        else:
                            token_text.append(token_text_ori[num])
                            num += 1
                    corpus.append(token_text)
        model = gensim.models.word2vec.Word2Vec(corpus, vector_size=300, min_count=1)
        word2index = {word: self.special_wordIdx[word] for word in self.special_wordIdx}
        new_word2index = {word: i + len(self.special_wordIdx) for i, word in enumerate(model.wv.index_to_key)}
        word2index.update(new_word2index)
        word2embedding = [[0] * 300] * len(self.special_wordIdx) + [model.wv[word] for word in new_word2index]
        mask4key = np.zeros(len(word2index))
        mask4movie = np.zeros(len(word2index))
        for i, word in enumerate(word2index):
            if word.lower() in self.key2index:
                mask4key[i] = 1
            if '@' in word:
                mask4movie[i] = 1
        np.save('data/mask4key.npy', mask4key)
        np.save('data/mask4movie.npy', mask4movie)
        json.dump(word2index, open('data/word2index_redial.json', 'w', encoding='utf-8'), ensure_ascii=False)
        np.save('data/word2vec_redial.npy', word2embedding)

    def data_process(self, is_finetune=False):
        """
        对数据集进行处理，包括填充、实体提取等操作
        """

        def padding_word2vev(sentence, max_length):
            """
            对文本进行填充和词向量化
            """
            input_ids = []
            concept_mask = []
            dbpedia_mask = []
            for word in sentence:
                input_ids.append(self.word2index.get(word, self.special_wordIdx['<unk>']))
                concept_mask.append(self.key2index.get(word.lower(), self.concept_min))
                if '@' in word:
                    try:
                        entity = self.id2entity[int(word[1:])]
                        id = self.entity2entityId[entity]
                    except:
                        id = self.entity_max
                    dbpedia_mask.append(id)
                else:
                    dbpedia_mask.append(self.entity_max)
            if len(input_ids) > max_length:
                return input_ids[-max_length:], max_length, concept_mask[-max_length:], dbpedia_mask[-max_length:]
            else:
                length = len(input_ids)
                return input_ids + (max_length - len(input_ids)) * [self.special_wordIdx['<pad>']], length, concept_mask + (max_length - len(input_ids)) * [self.concept_min], dbpedia_mask + (max_length - len(input_ids)) * [self.entity_max]

        data_set = []
        context_before = []
        for case in self.cases:
            if is_finetune and case['contexts'] == context_before:
                continue
            else:
                context_before = case['contexts']  # 区别
            contexts = sum((sen + ['<mood>', '<split>'] for sen in case['contexts'][-5:-1]), []) + case['contexts'][-1] + ['<mood>'] + ['<eos>']
            context_vector, context_length, concept_mask, dbpedia_mask = padding_word2vev(contexts, self.max_c_length)
            response_vector, r_length, _, _ = padding_word2vev(case['response'] + ['<eos>'], self.max_r_length)
            assert len(context_vector) == self.max_c_length
            assert len(concept_mask) == self.max_c_length
            assert len(dbpedia_mask) == self.max_c_length
            data_set.append([np.array(context_vector), np.array(response_vector), case['users'], case['user'], case['entity'], case['movie'], concept_mask, dbpedia_mask, case['rec']])
        return data_set

    def co_occurance_ext(self, cases):
        """
        提取数据集中的特定实体相关信息
        """
        keyword_sets = set(self.key2index.keys()) - self.stopwords
        movie_wordset = set()
        for case in cases:
            movie_words = []
            if case['rec'] == 1:
                for word in case['response']:
                    if '@' in word:
                        try:
                            num = self.entity2entityId[self.id2entity[int(word[1:])]]
                            movie_words.append(word)
                            movie_wordset.add(word)
                        except:
                            pass
            case['movie_words'] = movie_words
        new_edges = set()
        for case in cases:
            if len(case['movie_words']) > 0:
                before_set = set()
                after_set = set()
                co_set = set()
                for sen in case['contexts']:
                    for word in sen:
                        if word in keyword_sets:
                            before_set.add(word)
                        if word in movie_wordset:
                            after_set.add(word)
                for word in case['response']:
                    if word in keyword_sets:
                        co_set.add(word)

                for movie in case['movie_words']:
                    for word in list(before_set):
                        new_edges.add('co_before' + '\t' + movie + '\t' + word + '\n')
                    for word in list(co_set):
                        new_edges.add('co_occurance' + '\t' + movie + '\t' + word + '\n')
                    for word in case['movie_words']:
                        if word != movie:
                            new_edges.add('co_occurance' + '\t' + movie + '\t' + word + '\n')
                    for word in list(after_set):
                        new_edges.add('co_after' + '\t' + word + '\t' + movie + '\n')
                        for word_a in list(co_set):
                            new_edges.add('co_after' + '\t' + word + '\t' + word_a + '\n')
        f = open('data/co_occurance.txt', 'w', encoding='utf-8')
        f.writelines(list(new_edges))
        f.close()
        json.dump(list(movie_wordset), open('data/movie_word.json', 'w', encoding='utf-8'), ensure_ascii=False)

    def prepare_dbpedia_subkg(self):
        subkg = defaultdict(list)
        relation_cnt = defaultdict(int)
        relation_idx = {}
        datas = pkl.load(open(self.crs_data_path + '/subkg.pkl', 'rb'))
        for dbpediaId in datas:
            for (relation, related) in datas[dbpediaId]:
                relation_cnt[relation] += 1
        for relation in relation_cnt:
            if relation_cnt[relation] > 250:
                relation_idx[relation] = len(relation_idx)
        for dbpediaId in datas:
            for (relation, related) in datas[dbpediaId]:
                if relation in relation_idx:
                    assert dbpediaId < self.n_entity + 1075
                    assert related < self.n_entity + 1075
                    subkg[dbpediaId].append([related, relation_idx[relation] + 18])
        print(len(relation_idx))
        f = open(self.crs_data_path + '/train_data.jsonl', encoding='utf-8')
        for case in tqdm(f):
            lines = json.loads(case.strip())
            respondentWorkerId = lines['respondentWorkerId']
            initiatorWorkerId = lines['initiatorWorkerId']
            respondentQuestions = lines['respondentQuestions']
            initiatorQuestions = lines['initiatorQuestions']
            for movieId in respondentQuestions:
                rating = respondentQuestions[movieId]
                if self.id2entity[int(movieId)] is not None:
                    dbpediaId = self.entity2entityId[self.id2entity[int(movieId)]]
                    subkg[self.userId2userIdx[str(respondentWorkerId)] + self.n_entity].append([dbpediaId, rating["suggested"] * 9 + rating["seen"] * 3 + rating["liked"]])
            for movieId in initiatorQuestions:
                rating = initiatorQuestions[movieId]
                if self.id2entity[int(movieId)] is not None:
                    dbpediaId = self.entity2entityId[self.id2entity[int(movieId)]]
                    subkg[self.userId2userIdx[str(initiatorWorkerId)] + self.n_entity].append([dbpediaId, rating["suggested"] * 9 + rating["seen"] * 3 + rating["liked"]])

        json.dump(subkg, open(self.crs_data_path + '/dbpedia_subkg.jsonl', 'w', encoding='utf-8'), ensure_ascii=False)

    def __getitem__(self, index):
        context_vector, response_vector, users, user, entity, movie, concept_mask, dbpedia_mask, rec = self.datapre[index]

        entity_vec = np.zeros(self.n_entity)
        dbpedia_mentioned = np.zeros(50)
        point = 0
        for en in entity:
            entity_vec[en] = 1
            dbpedia_mentioned[point] = en
            point += 1

        concept_vector = np.zeros(self.n_concept)
        for con in concept_mask:
            if con != 0:
                concept_vector[con] = 1

        dbpedia_vector = np.zeros(self.n_entity)
        for db in dbpedia_mask:
            if db != 0:
                dbpedia_vector[db] = 1

        user_mentioned = np.zeros(100)
        user_vector = np.zeros(self.n_user)
        point = 0
        for us in users:
            if us != 0:
                user_vector[self.userId2userIdx[str(us)]] = 1
                user_mentioned[point] = self.userId2userIdx[str(us)]
                point += 1
        return torch.tensor(context_vector, dtype=torch.long), torch.tensor(response_vector, dtype=torch.long), entity_vec, torch.tensor(dbpedia_mentioned, dtype=torch.long), torch.tensor(user_mentioned, dtype=torch.long), movie, torch.tensor(concept_mask, dtype=torch.long), np.array(dbpedia_mask), concept_vector, dbpedia_vector, user_vector, rec

    def __len__(self):
        return len(self.datapre)
