import numpy as np
from multiprocessing import Pool
import mmap
import contextlib
class LR:
    def __init__(self, train_file_name, test_file_name, predict_result_file_name):
        self.train_file = train_file_name
        self.predict_file = test_file_name
        self.predict_result_file = predict_result_file_name
        self.max_iters = 37
        self.rate = 3
        self.feats = []
        self.labels = []
        self.feats_test = []
        self.labels_predict = []
        self.param_num = 0
        self.weight = []
        self.offset = 0

    def ReadData(self,n_processing,read_train_lines_set,trainNUM,read_test_lines_set):
        train_data = np.zeros((trainNUM,1001))
        read_lines_count = 0
        f = open(self.train_file, 'r')
        with contextlib.closing(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)) as m:
            m.seek(len(f.readline())*read_train_lines_set*n_processing,0)
            m.readline()
            while True:
                line = m.readline().strip()
                train_data[read_lines_count] = line.split(b',')
                read_lines_count += 1
                if read_lines_count == trainNUM:
                    break
        test_data = np.zeros((read_test_lines_set, 1000))
        read_lines_count = 0
        f = open(self.predict_file, 'r')
        with contextlib.closing(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)) as m:
            m.seek(6000 * n_processing * read_test_lines_set, 0)
            while True:
                test_data[read_lines_count] = m.readline().strip().split(b',')
                read_lines_count += 1
                if read_lines_count == read_test_lines_set:
                    break
        return train_data, test_data

    def loadData(self):
        pro_num = 8
        trf = open(self.train_file)
        trf.seek(0,2)
        read_train_lines_set = int(trf.tell()/6002/pro_num)
        trainNUM = 200
        trf.close

        tef = open(self.predict_file)
        tef.seek(0, 2)
        read_test_lines_set = int(tef.tell() / 6000 / pro_num)
        tef.close
        pool = Pool(processes = pro_num)
        job_result = []
        for i in range(pro_num):
            res = pool.apply_async(self.ReadData, (i, read_train_lines_set,trainNUM,read_test_lines_set))
            job_result.append(res)
        pool.close()
        pool.join()

        for tmp in job_result:
            temp_train, temp_test = tmp.get()
            self.feats.extend(temp_train)
            self.labels.extend(temp_train)
            self.feats_test.extend(temp_test)
        self.feats = np.array(self.feats)
        self.labels = self.feats[:,-1]
        self.feats = self.feats[:,:-1]

    def compute(self, recNum, param_num, feats, w):
        return 1 / (1 + np.exp(-np.dot(feats, w)))

    def predict(self,data):
        preval = self.compute(len(data), self.param_num, data, self.weight)
        self.labels_predict = (preval + 0.5).astype(np.int)
        f = open(self.predict_result_file, 'w')
        for i in range(len(self.labels_predict)):
            f.write(str(self.labels_predict[i]) + "\n")
        f.close()

    def train(self):
        self.loadData()
        recNum = len(self.feats)
        self.param_num = len(self.feats[0])
        self.weight = np.zeros((self.param_num,), dtype=np.float)
        for i in range(self.max_iters):
            preval = self.compute(recNum, self.param_num,
                                  self.feats, self.weight)
            err = self.labels - preval
            self.miniSGD(err=err)
            self.rate = self.rate*0.88

    def miniSGD(self,err):
        minibatch_size = 200
        down_pos = self.offset
        up_pos = self.offset+minibatch_size
        if up_pos > len(self.feats):
            down_pos = 0
            up_pos = down_pos + minibatch_size
        minibatch_feats = self.feats[range(down_pos,up_pos),:]
        minibatch_err = err[range(down_pos,up_pos)]
        delt_w = np.dot(minibatch_feats.T,minibatch_err)/minibatch_size
        self.weight += self.rate*delt_w
        if up_pos == len(self.feats):
            self.offset = 0
        else:
            self.offset = up_pos

if __name__ == "__main__":
    #文件路径
    train_file = "/data/train_data.txt"
    test_file = "/data/test_data.txt"
    predict_file = "/projects/student/result.txt"
    # start = time.time()
    lr = LR(train_file, test_file, predict_file)
    lr.train()
    lr.predict(lr.feats_test)
    # end = time.time()
    # print ('total time:',end - start)
