import numpy as np
V=[None, None, None, None, None, None, None, None, None, None]
#V=[0, 0, 0,0, 0, 0, 0, 0, 0, 0]
P=[[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]]
R=[[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]]

class Env: #환경에 대한 기술
    def __init__(self):
        self.V=V
        self.P=P
        self.R=R
        self.P[0][1]=0.5
        self.P[0][2]=0.5
        self.P[1][0]=0.4
        self.P[1][2]=0.3
        self.P[1][8]=0.3
        self.P[2][3]=0.8
        self.P[2][5]=0.2
        self.P[3][0]=0.6
        self.P[3][4]=0.4
        self.P[4][5]=0.5
        self.P[4][6]=0.5
        self.P[5][6]=0.4
        self.P[5][7]=0.3
        self.P[5][8]=0.3
        self.P[6][4]=0.2
        self.P[6][7]=0.4
        self.P[6][9]=0.4
        self.P[7][6]=0.2
        self.P[7][9]=0.8
        self.P[8][8]=0.4
        self.P[8][7]=0.6

        self.R[1][8]=-1
        self.R[1][2]=1
        self.R[3][0]=-1
        self.R[4][5]=1
        self.R[5][6]=1
        self.R[5][8]=-2
        self.R[5][7]=-1
        self.R[6][9]=10
        self.R[7][9]=2
    def show(self): #현상황 모니터링
        print('P:')
        for i in self.P:
            print(i)
        print('V:')
        for i in self.V:
            print(i)
    def PE(self): #정책 평가(policy evaluation)
        self.V[0]=self.P[0][1]*(R[0][1]+(0.9*self.V[1]))+self.P[0][2]*(R[0][2]+(0.9*self.V[2]))
        self.V[1]=self.P[1][0]*(R[1][0]+(0.9*self.V[0]))+self.P[1][2]*(R[1][2]+(0.9*self.V[2]))+self.P[1][8]*(R[1][8]+(0.9*self.V[8]))
        self.V[2]=self.P[2][3]*(R[2][3]+(0.9*self.V[3]))+self.P[2][5]*(R[2][5]+(0.9*self.V[5]))
        self.V[3]=self.P[3][0]*(R[3][0]+(0.9*self.V[0]))+self.P[3][4]*(R[3][4]+(0.9*self.V[4]))
        self.V[4]=self.P[4][5]*(R[4][5]+(0.9*self.V[5]))+self.P[4][6]*(R[4][6]+(0.9*self.V[6]))
        self.V[5]=self.P[5][6]*(R[5][6]+(0.9*self.V[6]))+self.P[5][7]*(R[5][7]+(0.9*self.V[7]))+self.P[5][8]*(R[5][8]+(0.9*self.V[8]))
        self.V[6]=self.P[6][4]*(R[6][4]+(0.9*self.V[4]))+self.P[6][7]*(R[6][7]+(0.9*self.V[7]))+self.P[6][9]*(R[6][9]+(0.9*self.V[9]))
        self.V[7]=self.P[7][6]*(R[7][6]+(0.9*self.V[6]))+self.P[7][9]*(R[7][9]+(0.9*self.V[9]))
        self.V[8]=self.P[8][7]*(R[8][7]+(0.9*self.V[7]))+self.P[8][8]*(R[8][8]+(0.9*self.V[8]))
        self.V[9]=0

    def monte_carlo(self, iters=1000, gamma=0.9, step_size=0.01):
        for _ in range(iters):
            state = np.random.randint(0, 8)
            episode = []
            while True:
                possible=[]
                for i, value in enumerate(self.P[state]):
                    if value > 0:
                        possible.append(i)
                next_state = np.random.choice(possible, 1)[0]
                reward = self.R[state][next_state]
                episode.append((state, next_state, reward))
                if next_state == 9:
                    state = next_state
                    episode.append((state, 0, 0))
                    break
                state = next_state
            Glist=[]
            G=0
            for i in episode[::-1]:
                st, ns, reward = i
                if st ==9:
                    continue
                G = gamma*G + self.R[st][ns]
                if self.V[st] == None:
                    self.V[st] = G*step_size
                else:
                    self.V[st] += step_size * (G - self.V[st])
                
            
            # for i in Glist:
            #     st, G = i
                # if st ==9:
                #     continue
                # print(st)
                # if self.V[st] == None:
                #     self.V[st] = G*step_size
                # else:
                #     self.V[st] += step_size*(G-self.V[st])
                
                
    def PI(self): #정책 개선(policy improvement)
        policy_stable = True  # 정책이 변경되었는지 여부를 추적합니다.
        for state in range(len(self.V) - 1):
            old_action = self.P[state]  # 현재 상태에서의 정책에 따른 행동을 찾습니다.
            # 가능한 모든 행동에 대해 가치를 계산하고 최선의 행동을 찾습니다.
            best_value = float('-inf')
            best_action = None

            for action, x in enumerate(self.P[state]):
                next_state=action
                value = self.R[state][action] + 0.9 * self.V[next_state]
                
                # 최선의 행동과 가치 업데이트
                if value > best_value:
                    best_value = value
                    best_action = action
            # 가치 함수를 이용하여 정책을 개선.
            # if best_action != np.argmax(old_action):
            #     policy_stable = False  # 정책이 변경되었음을 표시.
            #     self.P[state] = [0] * len(self.P[state])
            #     self.P[state][best_action] = 1
            if best_action is not None:  # 최고 가치를 갖는 행동이 있는 경우
                new_action = [0] * len(self.P[state])  # 새로운 정책을 초기화합니다.
                new_action[best_action] = 1  # 가장 가치가 높은 행동에 대한 확률을 1로 설정합니다.
                if new_action != old_action:  # 새로운 정책이 이전 정책과 다른 경우
                    policy_stable = False  # 정책이 변경되었음을 표시합니다.
                    self.P[state] = new_action  # 정책을 업데이트합니다.
        self.show()
        return policy_stable
    

policy_stable=False
itr_num=0
env1 = Env()#초반 값 변화를 따라가기위해 수기로 두차례 관찰
env1.show()
env1.monte_carlo(50, 0.9, 0.1)
env1.show()
env1.monte_carlo(1000, 0.9, 0.001)
env1.show()
env1.monte_carlo(1000, 0.9, 0.0001)
env1.show()
# env1.monte_carlo(10, 0.9, 0.001)
# env1.show()
# env1.monte_carlo(1000, 0.9, 0.0001)
# env1.show()
# env1.monte_carlo(1000, 0.9, 0.0001)
# env1.show()
# env1.monte_carlo(1000, 0.9, 0.0001)
# env1.show()
# env1.monte_carlo(1000, 0.9, 0.0001)
# env1.show()
# env1.monte_carlo(1000, 0.9, 0.000001)
# env1.show()
# env1.monte_carlo(100000000000000, 0.9, 0.0000001)
# env1.show()
