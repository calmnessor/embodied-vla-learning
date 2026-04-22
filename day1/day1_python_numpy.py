# 1. Python进阶 - OOP + 多进程

import multiprocessing as mp
import time 

class RobotTask:
    def __init__(self, name):
        self.name = name

    def run(self , delay = 1):
        print('%s is working...' % self.name)
        time.sleep(delay)
        print('%s is done.' % self.name)
#测试 oop
task = RobotTask('拾取苹果')
print(task.run())

# 多进程示例（VLA数据并行加载常用）
def worker(task_name):
    return RobotTask(task_name).run(2)
if __name__ == "__main__":
    tasks = ["拾取", "放置", "导航"]
    with mp.Pool(processes=3) as pool:
       results = pool.map(worker, tasks)
    print(results)
    
    
