import logging
import sys
import pandas as pd
import numpy as np
import random

logging.basicConfig(level=logging.DEBUG,
                    filename='./logs/CodeCraft-2019.log',
                    format='[%(asctime)s] %(levelname)s [%(funcName)s: %(filename)s, %(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filemode='a')


class Read_Data(object):
    def __init__(self,car_path,road_path,cross_path,answer_path):
        self.car_path = car_path
        self.road_path = road_path
        self.cross_path = cross_path
        self.answer_path = answer_path
        self.read_data()
        
    def read_data(self):
        with open(self.car_path,'r') as f:
            title = f.readline()
            title = title.strip('#(').strip(')\n').split(',')
            data = f.readlines()
            data = [i.strip('(').strip(')\n').split(',') for i in data]
            data = np.array([np.array(i,dtype=np.int16) for i in data])
            self.car_df = pd.DataFrame(columns=title,data=data)
        with open(self.road_path,'r') as f:
            title = f.readline()
            title = title.strip('#(').strip(')\n').split(',')
            data = f.readlines()
            data = [i.strip('(').strip(')\n').split(',') for i in data]
            data = np.array([np.array(i,dtype=np.int16) for i in data])
            self.road_df = pd.DataFrame(columns=title,data=data)
        with open(self.cross_path,'r') as f:
            title = f.readline()
            title = title.strip('#(').strip(')\n').split(',')
            data = f.readlines()
            data = [i.strip('(').strip(')\n').split(',') for i in data]
            data = np.array([np.array(i,dtype=np.int16) for i in data])
            self.cross_df = pd.DataFrame(columns=title,data=data)
        logging.info("car num is %d" % (self.car_df.shape[0]))
        logging.info("road num is %d" % (self.road_df.shape[0]))
        logging.info("cross num is %d" % (self.cross_df.shape[0]))

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
class Map_Create(object):
    def __init__(self,road_df, cross_df):
        self.road_df = road_df
        self.cross_df = cross_df
        #为了更好的索引节点，邻接矩阵大一点
        self.node_num = self.cross_df.shape[0]+1
        self.max_num = 100000
        #路径图，节点数据分别为 id length channel speed time
        self.map = np.full((self.node_num,self.node_num,5),self.max_num,dtype=int)
        self.init_map()
        self.id_idx = 0
        self.length_idx = 1
        self.channel_idx = 2
        self.speed_idx = 3
        self.time_idx = 4
        
        self.shortest_path = np.ndarray((self.node_num,self.node_num),dtype=object)
        self.find_path()
    def init_map(self):
        for i in self.road_df.index:
            road =self.road_df.loc[i,:]
            road_value = np.full((5,),self.max_num,dtype=int)
            road_value[0:4] = road.values[0:4]
            road_value[4] = int(road_value[1] / road_value[3])
            node_from = road.loc['from']
            node_to = road.loc['to']
            self.map[node_from][node_to] = road_value
            if road.loc['isDuplex'] == 1:
                self.map[node_to][node_from] = road_value
            logging.info("creat map ok")
    
    '''
    单源最短路径
    输入：当前节点
    输出：最短路径
    '''
    def dijkstra(self,node):
        d = np.full((self.node_num,),self.max_num) #存储节点间的最短距离
        pre = np.arange(0,self.node_num)      #存储最短路径的前驱
        vis = np.full((self.node_num,),0)     #是否访问标记
        
        d[node] = 0
        for i in range(1,self.node_num):
            u = -1
            min_num = self.max_num
            for j in range(1,self.node_num):
                if(vis[j]==0 and d[j]<min_num):
                    u = j
                    min_num = d[j]
            vis[u] = 1
            for v in range(1,self.node_num):
                du = d[u] + self.map[u][v][self.time_idx]
                if (vis[v]==0 and self.map[u][v][self.time_idx]!=self.max_num and du <d[v]):
                    d[v] = du
                    pre[v] = u
        return pre
    #存储所有节点之间的最短路径
    def find_path(self):
        #对所有节点作dijkstra
        for i in range(1,self.node_num):
            pre = self.dijkstra(i)
            for j in range(1,self.node_num):
                path = DFS(i, j, pre)
                self.shortest_path[i][j] = path
    '''
    绕路判断
    输入：u，v节点及当前u到v最短的路径(节点)
    输出，u，v节点之间替代的路径
    '''
    def detour(self, u, v, path):
        #这条路不能走了
        self.map_cpy = self.map.copy()
        for i in range(1, len(path)):
            self.map_cpy[path[i-1]][path[i]][self.time_idx] = self.max_num
        #采用dijkstra重新规划
        pre = self.dijkstra(u)
        now_path = DFS(u, v, pre)
        
        return now_path
class Schedul_Strate(object):
    def __init__(self, map, car_df, shortest_path, answer_path):
        self.map = map
        self.car_df = car_df.copy()
        self.shortest_path = shortest_path
        self.answer_path = answer_path
        
    '''
    车辆调度
    先来先服务，行驶最短路径
    '''
    def schedule(self):
        #按时间排序
        self.car_df = self.car_df.sort_values(by='planTime',axis=0)
        self.car_df.reset_index(inplace=True,drop=True)
        with open(self.answer_path,'w') as f:
            f.writelines('#(carId,StartTime,RoadId...)\n')
            #循环处理车辆并写入结果
            for i in self.car_df.index:
                car_id,node_from,node_to,speed,planTime = self.car_df.loc[i,:]
                planTime = random.randint(planTime,planTime+700)
                buffer = '('+str(car_id)+','+str(planTime)
                now_path = self.shortest_path[node_from][node_to]
                for i in range(1,len(now_path)):
                    now_node = now_path[i]
                    last_node = now_path[i-1]
                    road_id = self.map[last_node][now_node][0]
                    buffer += ',' + str(road_id)
                buffer += ')\n'
                f.writelines(buffer)
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
    Strate = Schedul_Strate(Map.map, Data.car_df, Map.shortest_path, answer_path)
    Strate.schedule()
if __name__ == "__main__":
    main()
#    car_path = './1-map-training-1/car.txt'
#    road_path = './1-map-training-1/road.txt'
#    cross_path = './1-map-training-1/cross.txt'
#    answer_path = './1-map-training-1/answer.txt'
#    
#    Data = Read_Data(car_path, road_path, cross_path, answer_path)
#    Map = Map_Create(Data.road_df, Data.cross_df)
#    Strate = Schedul_Strate(Map.map, Data.car_df, Map.shortest_path, answer_path)
#    Strate.schedule()