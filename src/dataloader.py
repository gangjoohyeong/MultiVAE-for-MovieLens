import os
import pandas as pd
from scipy import sparse
import numpy as np
from .utils import get_logger, logging_conf

logger = get_logger(logger_conf=logging_conf)

class Preprocess:
    def __init__(self, args):
        self.args = args
        
    def get_count(self, tp, id):
        playcount_groupbyid = tp[[id]].groupby(id)
        count = playcount_groupbyid.size()

        return count


    def filter_triplets(self, tp, min_uc=5, min_sc=0):
        if min_sc > 0:
            itemcount = self.get_count(tp, 'item')
            tp = tp[tp['item'].isin(itemcount.index[itemcount >= min_sc])]

        if min_uc > 0:
            usercount = self.get_count(tp, 'user')
            tp = tp[tp['user'].isin(usercount.index[usercount >= min_uc])]

        usercount, itemcount = self.get_count(tp, 'user'), self.get_count(tp, 'item')
        return tp, usercount, itemcount

    def split_train_test_proportion(self, args, data, test_prop=0.2):
        data_grouped_by_user = data.groupby('user')
        tr_list, te_list = list(), list()

        np.random.seed(args.seed)
        
        for _, group in data_grouped_by_user:
            # 각 user에 대한 item의 개수
            n_items_u = len(group)
            
            # 해당 user가 interaction한 item이 5 이상이면
            if n_items_u >= 5:
                idx = np.zeros(n_items_u, dtype='bool')
                
                # idx의 원소를 test 비율만큼 True로 변경
                idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

                tr_list.append(group[np.logical_not(idx)])
                te_list.append(group[idx])
            
            # 해당 user가 interaction한 item이 5 미만이면
            else:
                # 무조건 train에만 넣음
                tr_list.append(group)
        
        data_tr = pd.concat(tr_list)
        data_te = pd.concat(te_list)

        return data_tr, data_te

    def numerize(self, tp, profile2id, show2id):
        uid = tp['user'].apply(lambda x: profile2id[x])
        sid = tp['item'].apply(lambda x: show2id[x])
        return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])


    def load_data_from_file(self, args):
        DATA_DIR = args.data
        raw_data = pd.read_csv(os.path.join(DATA_DIR, 'train_ratings.csv'), header=0)

        # Filter Data
        raw_data, user_activity, item_popularity = self.filter_triplets(raw_data, min_uc=5, min_sc=0)
    
        # Shuffle User Indices
        
        unique_uid = user_activity.index
        np.random.seed(args.seed)
        idx_perm = np.random.permutation(unique_uid.size)
        unique_uid = unique_uid[idx_perm]

        n_users = unique_uid.size # 31360
        n_heldout_users = 3000


        # Split Train/Validation/Test User Indices
        # 31360 - 3000 - 3000 = 25360
        tr_users = unique_uid[:(n_users - n_heldout_users * 2)]

        # 3000
        vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]

        # 3000
        te_users = unique_uid[(n_users - n_heldout_users):]



        #주의: 데이터의 수가 아닌 사용자의 수입니다!
        logger.info(f"🧍 Train - Number of users: {len(tr_users)}")
        logger.info(f"🧍 Validation - Number of users: {len(vd_users)}")
        logger.info(f"🧍 Test - Number of users: {len(te_users)}")

        return raw_data, unique_uid, tr_users, vd_users, te_users
        
    def data_split(self, args, raw_data, unique_uid, tr_users, vd_users, te_users):
        
        # 훈련에 사용할 데이터 [full or split]
        if args.mode == 'submission':
            train_plays = raw_data.loc[raw_data['user'].isin(unique_uid)]
        elif args.mode == 'tuning':
            train_plays = raw_data.loc[raw_data['user'].isin(tr_users)]
        

        unique_sid = pd.unique(train_plays['item'])

        show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
        profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

        id2show = {v:k for k,v in show2id.items()}
        id2profile = {v:k for k,v in profile2id.items()}
        
        DATA_DIR = args.data
        pro_dir = os.path.join(DATA_DIR, 'pro_sg')

        if not os.path.exists(pro_dir):
            os.makedirs(pro_dir)

        with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
            for sid in unique_sid:
                f.write('%s\n' % sid)

        #Validation과 Test에는 input으로 사용될 tr 데이터와 정답을 확인하기 위한 te 데이터로 분리되었습니다.
        vad_plays = raw_data.loc[raw_data['user'].isin(vd_users)]
        vad_plays = vad_plays.loc[vad_plays['item'].isin(unique_sid)]
        vad_plays_tr, vad_plays_te = self.split_train_test_proportion(args, vad_plays)

        test_plays = raw_data.loc[raw_data['user'].isin(te_users)]
        test_plays = test_plays.loc[test_plays['item'].isin(unique_sid)]
        test_plays_tr, test_plays_te = self.split_train_test_proportion(args, test_plays)




        train_data = self.numerize(train_plays, profile2id, show2id)
        train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)


        vad_data_tr = self.numerize(vad_plays_tr, profile2id, show2id)
        vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)

        vad_data_te = self.numerize(vad_plays_te, profile2id, show2id)
        vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)

        test_data_tr = self.numerize(test_plays_tr, profile2id, show2id)
        test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)

        test_data_te = self.numerize(test_plays_te, profile2id, show2id)
        test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)

        return unique_sid, show2id, profile2id, id2show, id2profile

class Dataloader():
    '''
    Load Movielens dataset
    '''
    def __init__(self, path):
        
        self.pro_dir = os.path.join(path, 'pro_sg')
        assert os.path.exists(self.pro_dir), "Preprocessed files do not exist. Run data.py"

        self.n_items = self.load_n_items()
    
    def load_data(self, datatype='train'):
        if datatype == 'train':
            # train.csv 전체를 [uid x sid]로 불러오기
            return self._load_train_data()
        elif datatype == 'validation':
            # validation에서 tr(train)/te(test)로 분리한 것을 각각 불러오기
            return self._load_tr_te_data(datatype)
        elif datatype == 'test':
            # test에서 tr(train)/te(test)로 분리한 것을 각각 불러오기
            return self._load_tr_te_data(datatype)
        else:
            raise ValueError("datatype should be in [train, validation, test]")
        
    def load_n_items(self):
        unique_sid = list()
        with open(os.path.join(self.pro_dir, 'unique_sid.txt'), 'r') as f:
            for line in f:
                unique_sid.append(line.strip())
        n_items = len(unique_sid)
        return n_items
    
    def _load_train_data(self):
        path = os.path.join(self.pro_dir, 'train.csv')
        
        tp = pd.read_csv(path)
        n_users = tp['uid'].max() + 1

        rows, cols = tp['uid'], tp['sid']
        
        # csr_matrix 생성 [uid x sid]
        data = sparse.csr_matrix((np.ones_like(rows),
                                 (rows, cols)), dtype='float64',
                                 shape=(n_users, self.n_items))
        return data
    
    def _load_tr_te_data(self, datatype='test'):
        tr_path = os.path.join(self.pro_dir, '{}_tr.csv'.format(datatype))
        te_path = os.path.join(self.pro_dir, '{}_te.csv'.format(datatype))

        tp_tr = pd.read_csv(tr_path)
        tp_te = pd.read_csv(te_path)

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                    (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, self.n_items))
        data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                    (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, self.n_items))
        return data_tr, data_te