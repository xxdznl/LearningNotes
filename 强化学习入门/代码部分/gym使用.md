# gym安装

gym正常安装：[GitHub - openai/gym: A toolkit for developing and comparing reinforcement learning algorithms.](https://github.com/openai/gym)

------

**关于在jupyter notebook上报错pygame.error: no available video device的问题，**

参考[python - openAi-gym NameError - Stack Overflow](https://stackoverflow.com/questions/44150310/openai-gym-nameerror)

1. 首先运行下面命令

   ```py
   apt-get install -y xvfb python-opengl > /dev/null 2>&1
   pip install gym pyvirtualdisplay > /dev/null 2>&1
   ```

2. 在jupyter notebook里导入如下库

   ```python
   import gym
   import numpy as np
   import matplotlib.pyplot as plt
   from IPython import display as ipythondisplay
   from pyvirtualdisplay import Display
   ```

3. 启用虚拟display

   ```python
   display = Display(visible=0, size=(400, 300))
   display.start()
   ```

4. 最终运行示例代码

   ```python
   env = gym.make('CartPole-v0')
   for i_episode in range(20):
      observation = env.reset()
      for t in range(100):
         plt.imshow(env.render(mode='rgb_array'))# CHANGED
         ipythondisplay.clear_output(wait=True) # ADDED
         ipythondisplay.display(plt.gcf()) # ADDED
         print(observation)
         action = env.action_space.sample()
         observation, reward, done, info = env.step(action)
         if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
   ```

   对比之前的代码

   ```python
   import gym
   env = gym.make('CartPole-v0')
   for i_episode in range(20):
       observation = env.reset()
       for t in range(100):
           env.render()
           print(observation)
           action = env.action_space.sample()
           observation, reward, done, info = env.step(action)
           if done:
               print("Episode finished after {} timesteps".format(t+1))
               break
   ```

```python
import gym
import numpy as np
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
display = Display(visible=0, size=(400, 300))
display.start()


plt.imshow(env.render(mode='rgb_array'))# CHANGED
ipythondisplay.clear_output(wait=True) # ADDED
ipythondisplay.display(plt.gcf()) # ADDED
```

**缺点：加载图片非常耗时，远远不如pygame**

