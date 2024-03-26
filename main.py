import numpy as np
V=[0,0,0,0,0,0,0,0,0,0]
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
    def show(self): #현상황 모니터링
        print('P:')
        for i in self.P:
            print(i)
        print('V:')
        for i in self.V:
            print(i)
    def PI(self): #정책 개선(policy improvement)
        policy_stable = True  # 정책이 변경되었는지 여부를 추적합니다.
        for state in range(len(self.V) - 1):
            old_action = self.P[state][:]  # 현재 상태에서의 정책에 따른 행동을 찾습니다.
            # 가능한 모든 행동에 대해 가치를 계산하고 최선의 행동을 찾습니다.
            best_value = float('-inf')
            best_action = None
            for action, _ in enumerate(self.P[state]):
                next_state=action
                if next_state != 0:  # 가능한 행동인 경우
                    # 행동에 대한 가치 계산
                    value = self.R[state][action] + 0.9 * self.V[next_state]
                    
                    # 최선의 행동과 가치 업데이트
                    if value > best_value:
                        best_value = value
                        best_action = action
            # 가치 함수를 이용하여 정책을 개선.
            if best_action != np.argmax(old_action):
                policy_stable = False  # 정책이 변경되었음을 표시.
                self.P[state] = [0] * len(self.P[state])
                self.P[state][best_action] = 1
        self.show()
        return policy_stable
    
policy_stable=False
itr_num=0
env1 = Env()#초반 값 변화를 따라가기위해 수기로 두차례 관찰
env1.show()
env1.PE()
env1.show()
env1.PE()
env1.show()
while not policy_stable:
    delta=10000
    tmp=env1.V[:9]
    flag=0
    print('PE loop start')
    while flag<2 :#두차례 이상 개선되지 않으면 진행-현상황 최적으로 판단
        env1.PE()
        for i, (t, v) in enumerate(zip(tmp[:9], env1.V[:9])):
            tmp[i]=abs(v-t)
        delta=min(tmp)
        if delta<0.000001:#충분히 작은수
            flag+=1
        tmp=env1.V[:9]
        print(delta,'-----------')
    
    policy_stable=env1.PI()#정책 개선
    itr_num+=1
    print('PI', itr_num, "time")
    env1.show()
print(itr_num,"번 최적화 해서 나온 최적해")
env1.show()
