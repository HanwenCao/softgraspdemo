import socket
import pickle
import select
from ggcnn_ur.msg import Grasp
import copy as copy_module



def recieve():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ip_port = ('127.0.0.1', 9999)
        s.bind(ip_port)
        s.setblocking(0)
        readable = select.select([s], [], [], 1.5)[0]

        grasp = Grasp()

        if readable:
            data, client_addr = s.recvfrom(1024)         
            grasp = pickle.loads(data)          
            return grasp
        else:            
            return grasp


def get_data(grasp_new):
    grasp = recieve() # if no update, keep 0
    if grasp.quality > 0.0:
        grasp_new = copy_module.deepcopy(grasp) # if no update, keep the old data
    # print(grasp_new)
    return grasp_new




if __name__ == '__main__':
    grasp_new = Grasp()
    while True:        
        # grasp = recieve() # if no update, keep 0
        
        # if grasp.quality > 0.0:
        #     grasp_new = copy_module.deepcopy(grasp) # if no update, keep the old data
        # print(grasp_new)
        grasp_new = get_data(grasp_new)
        print(grasp_new)
    s.close()
