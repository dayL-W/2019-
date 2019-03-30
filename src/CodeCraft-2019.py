import logging
import sys
import pandas as pd
import numpy as np
import random
import queue
import math
import  time

logging.basicConfig(level=logging.DEBUG,
                    filename='./logs/CodeCraft-2019.log',
                    format='[%(asctime)s] %(levelname)s [%(funcName)s: %(filename)s, %(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filemode='a')

random.seed(2019)
class Read_Data(object):
    def __init__(self,car_path,road_path,cross_path,answer_path):
        self.car_path = car_path
        self.road_path = road_path
        self.cross_path = cross_path
        self.answer_path = answer_path
        self.read_data()
        self.preprocess()
        
    def read_data(self):
        with open(self.car_path,'r') as f:
            title = f.readline()
            title = title.strip('#(').strip(')\n').split(',')
            data = f.readlines()
            data = [i.strip('(').strip(')\n').split(',') for i in data]
            data = np.array([np.array(i,dtype=np.int64) for i in data])
            self.car_df = pd.DataFrame(columns=title,data=data)
        with open(self.road_path,'r') as f:
            title = f.readline()
            title = title.strip('#(').strip(')\n').split(',')
            data = f.readlines()
            data = [i.strip('(').strip(')\n').split(',') for i in data]
            data = np.array([np.array(i,dtype=np.int64) for i in data])
            self.road_df = pd.DataFrame(columns=title,data=data)
        with open(self.cross_path,'r') as f:
            title = f.readline()
            title = title.strip('#(').strip(')\n').split(',')
            data = f.readlines()
            data = [i.strip('(').strip(')\n').split(',') for i in data]
            data = np.array([np.array(i,dtype=np.int64) for i in data])
            self.cross_df = pd.DataFrame(columns=title,data=data)
        logging.info("car num is %d" % (self.car_df.shape[0]))
        logging.info("road num is %d" % (self.road_df.shape[0]))
        logging.info("cross num is %d" % (self.cross_df.shape[0]))
        pass
    '''
    数据预处理，把路口的id全部换成0到142
    '''
    def preprocess(self):
        self.num_to_node = self.cross_df.id.copy()
        self.node_to_num = pd.Series(index=self.num_to_node.values,data=self.num_to_node.index)
        
        self.cross_df['id_a'] = self.cross_df['id'].apply(lambda x:self.node_to_num[x])
        self.car_df['from_a'] = self.car_df['from'].apply(lambda x:self.node_to_num[x])
        self.car_df['to_a'] = self.car_df['to'].apply(lambda x:self.node_to_num[x])
        self.road_df['from_a'] = self.road_df['from'].apply(lambda x:self.node_to_num[x])
        self.road_df['to_a'] = self.road_df['to'].apply(lambda x:self.node_to_num[x])
        
'''
返回s到v的最短路径
放在类里面提示未定义DFS
'''
def DFS(s, v, pre):
    path = np.array([],dtype=int)
    if (s==v):
        path = np.append(path,v)
        return path
    t = DFS(s, pre[v], pre)
    path = np.append(path,t)
    path =np.append(path,v)
    return path

'''
返回前驱表示的所有的最短路径
'''
def get_paths(start, index, pre):
    child = []
    mid = []
    if index != start:
        for i in pre[index]:
            child = get_paths(start, i, pre)
            for j in child:
                j.append(index)
            if len(mid)==0:
                mid = child
            else:
                for t in child:
                    mid.append(t)
    else:
        mid.append([start])
    return mid
class Map_Create(object):
    def __init__(self,road_df, cross_df):
        self.road_df = road_df
        self.cross_df = cross_df
        #为了更好的索引节点，邻接矩阵大一点
        self.node_num = self.cross_df.shape[0]
        self.max_num = 100000
        #路径图，储存的地图分别有普通地图，高速路地图，低速路地图
        self.map = np.full((3,self.node_num,self.node_num,5),self.max_num,dtype=int)
        self.norm_map_idx = 0
        self.high_map_idx = 1
        self.low_map_idx = 2
        self.norm_path_idx = 0
        self.high_path_idx = 1
        self.low_path_idx = 2
        self.id_idx = 0
        self.length_idx = 1
        self.speed_idx = 2
        self.channel_idx = 3
        self.time_idx = 4
        
        self.init_map()
        self.init_high_way()
        self.init_low_way()
        self.path = np.ndarray((3,self.node_num,self.node_num),dtype=object)
        self.all_shortest_paths = np.ndarray((self.node_num,self.node_num),dtype=object)
#        self.find_path()
        self.find_all_path()
    
    def init_map(self):
        for i in self.road_df.index:
            road =self.road_df.loc[i,['id','length','speed','channel','from_a','to_a','isDuplex']]
            road_value = np.full((5,),self.max_num,dtype=int)
            road_value[0:4] = road.values[0:4]
            road_value[4] = math.ceil(road.length / road.speed)
            node_from = road.loc['from_a']
            node_to = road.loc['to_a']
            self.map[self.norm_map_idx][node_from][node_to] = road_value
            if road.loc['isDuplex'] == 1:
                self.map[self.norm_map_idx][node_to][node_from] = road_value
            logging.info("creat map ok")
        pass
   
    def BFS(self, u, map_idx):
        q = queue.Queue()
        self.inq[u] = 1
        q.put_nowait(u)
        while(q.empty() != True):
            u = q.get_nowait()
            for v in range(self.node_num):
                if self.inq[v] == 0 and self.map[map_idx][u][v][self.time_idx] != self.max_num:
                    q.put_nowait(v)
                    self.inq[v] = 1     
        pass
    '''
    广度遍历图
    返回连通分量的个数，以及每个连通分量其中一个节点
    '''
    def BFS_Trave(self,map_idx):
        self.inq = np.full((self.node_num,),0,dtype=int)
        BFS_num = 0
        node = np.array([],dtype=int)
        for u in range(self.node_num):
            if self.inq[u]==0:
                self.BFS(u,map_idx)
                BFS_num += 1
                node = np.append(node,u)
        return BFS_num, node
    '''
    创建高速路，即限速6,8的道路
    如果图不连通，打开某个节点的所有边
    '''
    def init_high_way(self):
        self.map[self.high_map_idx] = self.map[self.norm_map_idx].copy()
        road_value = np.full((5,),self.max_num,dtype=int)
        #遍历矩阵，如果限速小于6，则删除这条道路
        for i in range(self.node_num):
            for j in range(self.node_num):
                if self.map[self.norm_map_idx][i][j][self.speed_idx] != self.max_num and self.map[self.norm_map_idx][i][j][self.speed_idx] < 6:
                    self.map[self.high_map_idx][i][j] = road_value
        #得到当前的2连通分量
        BFS_num, node = self.BFS_Trave(self.high_map_idx)
        if BFS_num != 1:
            for i in node:
                self.map[self.high_map_idx][i,:,:] = self.map[self.norm_map_idx][i,:,:]
                self.map[self.high_map_idx][:,i,:] = self.map[self.norm_map_idx][:,i,:]
        pass
    '''
    创建低速路，即限速为4的道路
    如果块数过多，考虑划分4,6为低速路
    '''
    def init_low_way(self):
        self.map[self.low_map_idx] = self.map[self.norm_map_idx].copy()
        road_value = np.full((5,),self.max_num,dtype=int)
        #遍历矩阵，如果限速小于6，则删除这条道路
        for i in range(self.node_num):
            for j in range(self.node_num):
                if self.map[self.norm_map_idx][i][j][self.speed_idx] != self.max_num and self.map[self.norm_map_idx][i][j][self.speed_idx] > 6:
                    self.map[self.low_map_idx][i][j] = road_value
        #得到当前的2连通分量
        BFS_num, node = self.BFS_Trave(self.low_map_idx)
        if BFS_num != 1:
            for i in node:
                self.map[self.low_map_idx][i,:,:] = self.map[self.norm_map_idx][i,:,:]
                self.map[self.low_map_idx][:,i,:] = self.map[self.norm_map_idx][:,i,:]
        '''
        放开21的边
        '''
        self.map[self.low_map_idx][21,:,:] = self.map[self.norm_map_idx][21,:,:]
        self.map[self.low_map_idx][:,21,:] = self.map[self.norm_map_idx][:,21,:]
        
        pass
    '''
    单源最短路径
    输入：当前节点
    输出：最短路径
    '''
    def dijkstra(self,node,map_idx):
        d = np.full((self.node_num,),self.max_num) #存储节点间的最短距离
        pre = np.arange(0,self.node_num)      #存储最短路径的前驱
        vis = np.full((self.node_num,),0)     #是否访问标记
        
        d[node] = 0
        for i in range(self.node_num):
            u = -1
            min_num = self.max_num
            for j in range(self.node_num):
                if(vis[j]==0 and d[j]<min_num):
                    u = j
                    min_num = d[j]
            vis[u] = 1
            for v in range(self.node_num):
                du = d[u] + self.map[map_idx][u][v][self.time_idx]
                if (vis[v]==0 and self.map[map_idx][u][v][self.time_idx]!=self.max_num and du <d[v]):
                    d[v] = du
                    pre[v] = u
        return pre
    '''
    单源最短路径
    输入：当前节点
    输出：所有的最短路径，所有的最短路径只有一条？？
    '''
    def dijkstra_all(self,node,map_idx):
        d = np.full((self.node_num,),self.max_num) #存储节点间的最短距离
        pre = []                                   #存储所有最短路径的前驱
        vis = np.full((self.node_num,),0)          #是否访问标记
        for i in range(self.node_num):
            pre.append([i])
        d[node] = 0
        for i in range(self.node_num):
            u = -1
            min_num = self.max_num
            for j in range(self.node_num):
                if(vis[j]==0 and d[j]<min_num):
                    u = j
                    min_num = d[j]
            vis[u] = 1
            for v in range(self.node_num):
                du = d[u] + self.map[map_idx][u][v][self.time_idx]
                if (vis[v]==0 and self.map[map_idx][u][v][self.time_idx]!=self.max_num):
                    if du < d[v]:
                        d[v] = du
                        pre[v] = []
                        pre[v].append(u)
                    elif du <= (d[v]+3):
                        pre[v].append(u)
        return pre
    #存储所有节点之间的最短路径
    def find_path(self):
        #对所有节点作dijkstra
#        for path_idx in range(3):
        path_idx = 0
        for i in range(self.node_num):
            pre = self.dijkstra(i,path_idx)
            for j in range(self.node_num):
                path = DFS(i, j, pre)
                self.path[path_idx][i][j] = path
        pass
    #存储所有节点之间的所有最短路径
    def find_all_path(self):
        for i in range(self.node_num):
            pre = self.dijkstra_all(i,self.norm_map_idx)
            for j in range(self.node_num):
                path = get_paths(i, j, pre)
                self.all_shortest_paths[i][j] = path
        pass
    '''
    绕路判断
    输入：u，v节点及当前u到v最短的路径(节点)
    输出，u，v节点之间替代的路径
    '''
    def detour(self, u, v, path):
        #这条路不能走了
        self.map_cpy = self.map.copy()
        for i in range( len(path)):
            self.map_cpy[path[i-1]][path[i]][self.time_idx] = self.max_num
        #采用dijkstra重新规划
        pre = self.dijkstra(u)
        now_path = DFS(u, v, pre)
        
        return now_path
class Schedul_Strate(object):
    def __init__(self, map_, car_df, path, all_shortest_paths, answer_path):
        self.norm_map_idx = 0
        self.high_map_idx = 1
        self.low_map_idx = 2
        self.id_idx = 0
        self.length_idx = 1
        self.speed_idx = 2
        self.channel_idx = 3
        self.time_idx = 4
        self.map = map_
        self.car_df = car_df.copy()
        self.path = path
        self.all_shortest_paths = all_shortest_paths
        self.answer_path = answer_path
        self.car_time = pd.DataFrame(columns=['id','arrive','use','add','begin','end'],dtype=np.int32)
        
    '''
    车辆调度
    先来先服务，行驶最短路径
    '''
    def fifo_schedule(self):
        '''
        道路5039，有0.2的概率被替换成5033,5032,5038,5044,5045或者反向
        '''
        #按时间排序
        self.car_df = self.car_df.sort_values(by='planTime',axis=0)
        self.car_df.reset_index(inplace=True,drop=True)
        with open(self.answer_path,'w') as f:
            f.writelines('#(carId,StartTime,RoadId...)\n')
            #循环处理车辆并写入结果
            road_use = np.full((105,),0,dtype=int)
            for car_idx in self.car_df.index:
                car_ser = pd.Series(index=['id','arrive','use','add', 'begin','end'],dtype=np.int32,data=0)
                use_time = 0
                car_id,node_from,node_to,car_speed,planTime = self.car_df.loc[car_idx,:]
                if car_speed >= 6 :
                    map_idx = self.high_map_idx
                else:
                    map_idx = self.norm_map_idx
                start_time = random.randint(planTime,planTime+598)
                buffer = '('+str(car_id)+','+str(start_time)

                now_path = self.path[map_idx][node_from][node_to]
                for i in range(1,len(now_path)):
                    now_node = now_path[i]
                    last_node = now_path[i-1]
                    road_id = self.map[map_idx][last_node][now_node][0]
                    buffer += ',' + str(road_id)
                    
                    road_speed = self.map[map_idx][last_node][now_node][self.speed_idx]
                    road_length = self.map[map_idx][last_node][now_node][self.length_idx]
                    use_time += math.ceil(min(car_speed, road_speed) / road_length)
                car_ser.loc['id'] = car_id
                car_ser.loc['arrive'] = planTime
                car_ser.loc['use'] = use_time
                car_ser.loc['add'] = int(planTime + use_time)
                self.car_time.loc[car_idx] = car_ser

                buffer += ')\n'
                f.writelines(buffer)
        self.car_time.sort_values(by=['add','arrive'],inplace=True)
        self.car_time.reset_index(drop=True,inplace=True)
        return road_use
    '''
    车辆调度
    先来先服务，随机选择一条最短的路径
    '''
    def rdom_path_schedule(self):
        #按时间排序
        self.car_df = self.car_df.sort_values(by='planTime',axis=0)
        self.car_df.reset_index(inplace=True,drop=True)
        with open(self.answer_path,'w') as f:
            f.writelines('#(carId,StartTime,RoadId...)\n')
            #循环处理车辆并写入结果
#            road_use = np.full((105,),0,dtype=int)
            for car_idx in self.car_df.index:
                car_ser = pd.Series(index=['id','arrive','use','add', 'begin','end'],dtype=np.int32,data=0)
                use_time = 0
                car_id,node_from,node_to,car_speed,planTime = self.car_df.loc[car_idx,['id','from_a','to_a','speed','planTime']]
                map_idx = self.norm_map_idx
                start_time = random.randint(planTime,planTime+3000)
                buffer = '('+str(car_id)+','+str(start_time)
                
                path_num = len(self.all_shortest_paths[node_from][node_to])
                path_idx = random.randint(0, path_num-1)
                now_path = self.all_shortest_paths[node_from][node_to][path_idx]
                for i in range(1,len(now_path)):
                    now_node = now_path[i]
                    last_node = now_path[i-1]
                    road_id = self.map[map_idx][last_node][now_node][0]
#                    road_use[road_id-5000] += 1
                    buffer += ',' + str(road_id)
                    
                    road_speed = self.map[map_idx][last_node][now_node][self.speed_idx]
                    road_length = self.map[map_idx][last_node][now_node][self.length_idx]
                    use_time += math.ceil(min(car_speed, road_speed) / road_length)
                car_ser.loc['id'] = car_id
                car_ser.loc['arrive'] = planTime
                car_ser.loc['use'] = use_time
                car_ser.loc['add'] = int(planTime + use_time)
                self.car_time.loc[car_idx] = car_ser
                
                buffer += ')\n'
                f.writelines(buffer)
#        return road_use
    '''
    限制道路网上车的数量
    '''
    def restrict_car(self):
        self.car_max_num = 150
        now_time = 1
        car_queue = queue.Queue(maxsize=self.car_max_num)
        #对所有的车辆
        for idx in self.car_time.index:
            if car_queue.full() == True:
                #把队列拿出来重新按照end_time排序
                qu_df = pd.DataFrame(columns=['id','arrive','use','add','begin','end'])
                cnt = 0
                while car_queue.empty() == False:
                    qu_df.loc[cnt,:] = car_queue.get_nowait()
                    cnt += 1
                qu_df.sort_values(by=['end'],inplace=True)
                for x in qu_df.index:
                    now_car = qu_df.loc[x]
                    car_queue.put_nowait(now_car)
                now_car = car_queue.get_nowait()
                now_time = now_car.end
                while car_queue.empty() == False:
                    now_car = car_queue.get_nowait()
                    tmp = now_car.end
                    if tmp != now_time:
                        break
            if car_queue.full() == False:
                now_car = self.car_time.loc[idx]
                now_car.begin = now_time
                now_car.end = now_time + now_car.use
                car_queue.put_nowait(now_car)
    '''
    生成车辆限行的答案
    '''
    def restrict_schedule(self):
        #打开一个临时文件
        buffer = ''
        with open(self.answer_path, 'r') as f:
            buffer = f.readline()
            for line in f:
                tmp = ''
                car_path = line.split(',')
                car_id = int(car_path[0].strip('('))
                begin = int(self.car_time.loc[self.car_time.id==car_id,'begin'])
                car_path[1] = str(begin)
                for i in car_path:
                    tmp += ','+i
                tmp = tmp.strip(',')
                buffer += tmp
        with open(self.answer_path, 'w') as f:
            f.write(buffer)
                
def main():
    if len(sys.argv) != 5:
        logging.info('please input args: car_path, road_path, cross_path, answerPath')
        exit(1)
    
    car_path = sys.argv[1]
    road_path = sys.argv[2]
    cross_path = sys.argv[3]
    answer_path = sys.argv[4]
    
    logging.info("car_path is %s" % (car_path))
    logging.info("road_path is %s" % (road_path))
    logging.info("cross_path is %s" % (cross_path))
    logging.info("answer_path is %s" % (answer_path))
    
    # to read input file
    Data = Read_Data(car_path, road_path, cross_path, answer_path)
    # process
    Map = Map_Create(Data.road_df, Data.cross_df)
    # to write output file
    Strate = Schedul_Strate(Map.map, Data.car_df, Map.path, Map.all_shortest_paths, answer_path)
    Strate.rdom_path_schedule()
#    Strate.restrict_car()
#    Strate.restrict_schedule()
if __name__ == "__main__":
#    main()
    
    time0 = time.time()
    car_path = './1-map-exam-2/car.txt'
    road_path = './1-map-exam-2/road.txt'
    cross_path = './1-map-exam-2/cross.txt'
    answer_path = './1-map-exam-2/answer.txt'
    
    Data = Read_Data(car_path, road_path, cross_path, answer_path)
    Map = Map_Create(Data.road_df, Data.cross_df)
    Strate = Schedul_Strate(Map.map, Data.car_df, Map.path, Map.all_shortest_paths, answer_path)
#    Strate.rdom_path_schedule()
    time1 = time.time()
    print('use:',time1-time0)
#    Strate.restrict_car()
#    Strate.restrict_schedule()
    