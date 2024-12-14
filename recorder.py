{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/EmaSuriano/pytorch-demo/blob/main/recorder.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "WrxRn2N8IBu3"
      },
      "source": [
        "## DeepQN Tutorial\n",
        "\n",
        "Based on [Reinforcement Learning (DQN) Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "id": "Mt10h8lYIBu4",
        "outputId": "44ae7abf-6336-42ea-9397-d3669059ff08",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Torch is properly installed!\n"
          ]
        }
      ],
      "source": [
        "import torch\n",
        "\n",
        "# https://pytorch.org/get-started/locally/\n",
        "print(\"Torch is properly installed!\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "metadata": {
        "id": "-7mUudnUIBu5",
        "outputId": "55c6673a-1691-48de-a86b-c7db65592638",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: matplotlib in /usr/local/lib/python3.10/dist-packages (3.8.0)\n",
            "Requirement already satisfied: opencv-python in /usr/local/lib/python3.10/dist-packages (4.10.0.84)\n",
            "Requirement already satisfied: ale-py in /usr/local/lib/python3.10/dist-packages (0.10.1)\n",
            "Requirement already satisfied: gymnasium[classic_control] in /usr/local/lib/python3.10/dist-packages (1.0.0)\n",
            "Requirement already satisfied: imageio[ffmpeg] in /usr/local/lib/python3.10/dist-packages (2.36.1)\n",
            "Requirement already satisfied: numpy>=1.21.0 in /usr/local/lib/python3.10/dist-packages (from gymnasium[classic_control]) (1.26.4)\n",
            "Requirement already satisfied: cloudpickle>=1.2.0 in /usr/local/lib/python3.10/dist-packages (from gymnasium[classic_control]) (3.1.0)\n",
            "Requirement already satisfied: typing-extensions>=4.3.0 in /usr/local/lib/python3.10/dist-packages (from gymnasium[classic_control]) (4.12.2)\n",
            "Requirement already satisfied: farama-notifications>=0.0.1 in /usr/local/lib/python3.10/dist-packages (from gymnasium[classic_control]) (0.0.4)\n",
            "Requirement already satisfied: pygame>=2.1.3 in /usr/local/lib/python3.10/dist-packages (from gymnasium[classic_control]) (2.6.1)\n",
            "Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (1.3.1)\n",
            "Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (0.12.1)\n",
            "Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (4.55.3)\n",
            "Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (1.4.7)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (24.2)\n",
            "Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (11.0.0)\n",
            "Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (3.2.0)\n",
            "Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from matplotlib) (2.8.2)\n",
            "Requirement already satisfied: imageio-ffmpeg in /usr/local/lib/python3.10/dist-packages (from imageio[ffmpeg]) (0.5.1)\n",
            "Requirement already satisfied: psutil in /usr/local/lib/python3.10/dist-packages (from imageio[ffmpeg]) (5.9.5)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.7->matplotlib) (1.17.0)\n",
            "Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from imageio-ffmpeg->imageio[ffmpeg]) (75.1.0)\n"
          ]
        }
      ],
      "source": [
        "%pip install gymnasium[classic_control] gymnasium[atari] matplotlib opencv-python imageio[ffmpeg] ale-py"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "\"\"\"\n",
        "Adapted from\n",
        "https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/atari_wrappers.py\n",
        "to store video\n",
        "\"\"\"\n",
        "\n",
        "from typing import Dict, SupportsFloat, Tuple, Any\n",
        "\n",
        "import gymnasium as gym\n",
        "import numpy as np\n",
        "from gymnasium import spaces\n",
        "import cv2\n",
        "\n",
        "AtariResetReturn = Tuple[np.ndarray, Dict[str, Any]]\n",
        "AtariStepReturn = Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]\n",
        "\n",
        "\n",
        "def process_image(\n",
        "    image, crop_size=(34, 194, 0, 160), target_size=(84, 84), normalize=True\n",
        "):\n",
        "    \"\"\"\n",
        "    Grayscale, crop and resize image\n",
        "\n",
        "    Input\n",
        "    - image: shape(h,w,c),(210,160,3)\n",
        "    - crop_size: shape(min_h,max_h,min_w,max_w)\n",
        "    - target_size: (h,w)\n",
        "    - normalize: [0,255] -> [0,1]\n",
        "\n",
        "    Output\n",
        "    - shape(84,84)\n",
        "    \"\"\"\n",
        "    frame = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # To grayscale\n",
        "    frame = frame[crop_size[0] : crop_size[1], crop_size[2] : crop_size[3]]\n",
        "    frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)  # Resize\n",
        "    if normalize:\n",
        "        return frame.astype(np.float32) / 255  # normalize\n",
        "    else:\n",
        "        return frame\n",
        "\n",
        "\n",
        "class StickyActionEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):\n",
        "    \"\"\"\n",
        "    Sticky action.\n",
        "\n",
        "    Paper: https://arxiv.org/abs/1709.06009\n",
        "    Official implementation: https://github.com/mgbellemare/Arcade-Learning-Environment\n",
        "\n",
        "    :param env: Environment to wrap\n",
        "    :param action_repeat_probability: Probability of repeating the last action\n",
        "    \"\"\"\n",
        "\n",
        "    def __init__(self, env: gym.Env, action_repeat_probability: float) -> None:\n",
        "        super().__init__(env)\n",
        "        self.action_repeat_probability = action_repeat_probability\n",
        "        assert env.unwrapped.get_action_meanings()[0] == \"NOOP\"  # type: ignore[attr-defined]\n",
        "\n",
        "    def reset(self, **kwargs) -> AtariResetReturn:\n",
        "        self._sticky_action = 0  # NOOP\n",
        "        return self.env.reset(**kwargs)\n",
        "\n",
        "    def step(self, action: int) -> AtariStepReturn:\n",
        "        if self.np_random.random() >= self.action_repeat_probability:\n",
        "            self._sticky_action = action\n",
        "        return self.env.step(self._sticky_action)\n",
        "\n",
        "\n",
        "class NoopResetEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):\n",
        "    \"\"\"\n",
        "    Sample initial states by taking random number of no-ops on reset.\n",
        "    No-op is assumed to be action 0.\n",
        "\n",
        "    :param env: Environment to wrap\n",
        "    :param noop_max: Maximum value of no-ops to run\n",
        "    \"\"\"\n",
        "\n",
        "    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:\n",
        "        super().__init__(env)\n",
        "        self.noop_max = noop_max\n",
        "        self.override_num_noops = None\n",
        "        self.noop_action = 0\n",
        "        assert env.unwrapped.get_action_meanings()[0] == \"NOOP\"  # type: ignore[attr-defined]\n",
        "\n",
        "    def reset(self, **kwargs) -> AtariResetReturn:\n",
        "        self.env.reset(**kwargs)\n",
        "        if self.override_num_noops is not None:\n",
        "            noops = self.override_num_noops\n",
        "        else:\n",
        "            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)\n",
        "        assert noops > 0\n",
        "        obs = np.zeros(0)\n",
        "        info: Dict = {}\n",
        "        for _ in range(noops):\n",
        "            obs, _, terminated, truncated, info = self.env.step(self.noop_action)\n",
        "            if terminated or truncated:\n",
        "                obs, info = self.env.reset(**kwargs)\n",
        "        return obs, info\n",
        "\n",
        "\n",
        "class FireResetEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):\n",
        "    \"\"\"\n",
        "    Take action on reset for environments that are fixed until firing.\n",
        "\n",
        "    :param env: Environment to wrap\n",
        "    \"\"\"\n",
        "\n",
        "    def __init__(self, env: gym.Env) -> None:\n",
        "        super().__init__(env)\n",
        "        assert env.unwrapped.get_action_meanings()[1] == \"FIRE\"  # type: ignore[attr-defined]\n",
        "        assert len(env.unwrapped.get_action_meanings()) >= 3  # type: ignore[attr-defined]\n",
        "\n",
        "    def reset(self, **kwargs) -> AtariResetReturn:\n",
        "        self.env.reset(**kwargs)\n",
        "        obs, _, terminated, truncated, info = self.env.step(1)\n",
        "        if terminated or truncated:\n",
        "            self.env.reset(**kwargs)\n",
        "        obs, _, terminated, truncated, info = self.env.step(2)\n",
        "        if terminated or truncated:\n",
        "            self.env.reset(**kwargs)\n",
        "        return obs, info\n",
        "\n",
        "\n",
        "class EpisodicLifeEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):\n",
        "    \"\"\"\n",
        "    Make end-of-life == end-of-episode, but only reset on true game over.\n",
        "    Done by DeepMind for the DQN and co. since it helps value estimation.\n",
        "\n",
        "    :param env: Environment to wrap\n",
        "    \"\"\"\n",
        "\n",
        "    def __init__(self, env: gym.Env) -> None:\n",
        "        super().__init__(env)\n",
        "        self.lives = 0\n",
        "        self.was_real_done = True\n",
        "\n",
        "    def step(self, action: int) -> AtariStepReturn:\n",
        "        obs, reward, terminated, truncated, info = self.env.step(action)\n",
        "        self.was_real_done = terminated or truncated\n",
        "\n",
        "        # check current lives, make loss of life terminal,\n",
        "        # then update lives to handle bonus lives\n",
        "        lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]\n",
        "        if 0 < lives < self.lives:\n",
        "            # for Qbert sometimes we stay in lives == 0 condition for a few frames\n",
        "            # so its important to keep lives > 0, so that we only reset once\n",
        "            # the environment advertises done.\n",
        "            terminated = True\n",
        "\n",
        "        self.lives = lives\n",
        "        return obs, reward, terminated, truncated, info\n",
        "\n",
        "    def reset(self, **kwargs) -> AtariResetReturn:\n",
        "        \"\"\"\n",
        "        Calls the Gym environment reset, only when lives are exhausted.\n",
        "        This way all states are still reachable even though lives are episodic,\n",
        "        and the learner need not know about any of this behind-the-scenes.\n",
        "\n",
        "        :param kwargs: Extra keywords passed to env.reset() call\n",
        "        :return: the first observation of the environment\n",
        "        \"\"\"\n",
        "        if self.was_real_done:\n",
        "            obs, info = self.env.reset(**kwargs)\n",
        "        else:\n",
        "            # no-op step to advance from terminal/lost life state\n",
        "            obs, _, terminated, truncated, info = self.env.step(0)\n",
        "\n",
        "            # The no-op step can lead to a game over, so we need to check it again\n",
        "            # to see if we should reset the environment and avoid the\n",
        "            # monitor.py `RuntimeError: Tried to step environment that needs reset`\n",
        "            if terminated or truncated:\n",
        "                obs, info = self.env.reset(**kwargs)\n",
        "\n",
        "        self.lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]\n",
        "        return obs, info\n",
        "\n",
        "\n",
        "class MaxAndSkipEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):\n",
        "    \"\"\"\n",
        "    Return only every ``skip``-th frame (frameskipping)\n",
        "    and return the max between the two last frames.\n",
        "\n",
        "    :param env: Environment to wrap\n",
        "    :param skip: Number of ``skip``-th frame\n",
        "        The same action will be taken ``skip`` times.\n",
        "    \"\"\"\n",
        "\n",
        "    def __init__(self, env: gym.Env, skip: int = 4) -> None:\n",
        "        super().__init__(env)\n",
        "        # most recent raw observations (for max pooling across time steps)\n",
        "        assert (\n",
        "            env.observation_space.dtype is not None\n",
        "        ), \"No dtype specified for the observation space\"\n",
        "        assert (\n",
        "            env.observation_space.shape is not None\n",
        "        ), \"No shape defined for the observation space\"\n",
        "        self._obs_buffer = np.zeros(\n",
        "            (2, *env.observation_space.shape), dtype=env.observation_space.dtype\n",
        "        )\n",
        "        self._skip = skip\n",
        "\n",
        "    def step(self, action: int) -> AtariStepReturn:\n",
        "        \"\"\"\n",
        "        Step the environment with the given action\n",
        "        Repeat action, sum reward, and max over last observations.\n",
        "\n",
        "        :param action: the action\n",
        "        :return: observation, reward, terminated, truncated, information\n",
        "        \"\"\"\n",
        "        total_reward = 0.0\n",
        "        terminated = truncated = False\n",
        "        for i in range(self._skip):\n",
        "            obs, reward, terminated, truncated, info = self.env.step(action)\n",
        "            done = terminated or truncated\n",
        "            if i == self._skip - 2:\n",
        "                self._obs_buffer[0] = obs\n",
        "            if i == self._skip - 1:\n",
        "                self._obs_buffer[1] = obs\n",
        "            total_reward += float(reward)\n",
        "            if done:\n",
        "                break\n",
        "        # Note that the observation on the done=True frame\n",
        "        # doesn't matter\n",
        "        max_frame = self._obs_buffer.max(axis=0)\n",
        "\n",
        "        return max_frame, total_reward, terminated, truncated, info\n",
        "\n",
        "\n",
        "class ClipRewardEnv(gym.RewardWrapper):\n",
        "    \"\"\"\n",
        "    Clip the reward to {+1, 0, -1} by its sign.\n",
        "\n",
        "    :param env: Environment to wrap\n",
        "    \"\"\"\n",
        "\n",
        "    def __init__(self, env: gym.Env) -> None:\n",
        "        super().__init__(env)\n",
        "\n",
        "    def reward(self, reward: SupportsFloat) -> float:\n",
        "        \"\"\"\n",
        "        Bin reward to {+1, 0, -1} by its sign.\n",
        "\n",
        "        :param reward:\n",
        "        :return:\n",
        "        \"\"\"\n",
        "        return np.sign(float(reward))\n",
        "\n",
        "\n",
        "class WarpFrame(gym.ObservationWrapper[np.ndarray, int, np.ndarray]):\n",
        "    \"\"\"\n",
        "    Convert to grayscale and warp frames to 84x84 (default)\n",
        "    as done in the Nature paper and later work.\n",
        "\n",
        "    :param env: Environment to wrap\n",
        "    :param width: New frame width\n",
        "    :param height: New frame height\n",
        "    \"\"\"\n",
        "\n",
        "    def __init__(\n",
        "        self,\n",
        "        env: gym.Env,\n",
        "        width: int = 84,\n",
        "        height: int = 84,\n",
        "        normalize: bool = True,\n",
        "        video=None,\n",
        "    ) -> None:\n",
        "        super().__init__(env)\n",
        "        self.width = width\n",
        "        self.height = height\n",
        "        self.normalize = normalize\n",
        "        self.video = video\n",
        "        assert isinstance(\n",
        "            env.observation_space, spaces.Box\n",
        "        ), f\"Expected Box space, got {env.observation_space}\"\n",
        "\n",
        "        self.observation_space = spaces.Box(\n",
        "            low=0,\n",
        "            high=255,\n",
        "            shape=(self.height, self.width, 1),\n",
        "            dtype=env.observation_space.dtype,  # type: ignore[arg-type]\n",
        "        )\n",
        "\n",
        "    def observation(self, frame: np.ndarray) -> np.ndarray:\n",
        "        \"\"\"\n",
        "        returns the current observation from a frame\n",
        "\n",
        "        :param frame: environment frame\n",
        "        :return: the observation shape(84,84) by default\n",
        "        \"\"\"\n",
        "        if self.video is not None:\n",
        "            self.video.record(frame)\n",
        "        return process_image(\n",
        "            frame, target_size=(self.width, self.height), normalize=self.normalize\n",
        "        )\n",
        "\n",
        "\n",
        "class AtariWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):\n",
        "    \"\"\"\n",
        "    Atari 2600 preprocessings\n",
        "\n",
        "    Specifically:\n",
        "\n",
        "    * Noop reset: obtain initial state by taking random number of no-ops on reset.\n",
        "    * Frame skipping: 4 by default\n",
        "    * Max-pooling: most recent two observations\n",
        "    * Termination signal when a life is lost.\n",
        "    * Resize to a square image: 84x84 by default\n",
        "    * Grayscale observation\n",
        "    * Clip reward to {-1, 0, 1}\n",
        "    * Sticky actions: disabled by default\n",
        "    * Normalize the rgb output to [0,1]\n",
        "\n",
        "    See https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/\n",
        "    for a visual explanation.\n",
        "\n",
        "    .. warning::\n",
        "        Use this wrapper only with Atari v4 without frame skip: ``env_id = \"*NoFrameskip-v4\"``.\n",
        "\n",
        "    :param env: Environment to wrap\n",
        "    :param noop_max: Max number of no-ops\n",
        "    :param frame_skip: Frequency at which the agent experiences the game.\n",
        "        This correspond to repeating the action ``frame_skip`` times.\n",
        "    :param screen_size: Resize Atari frame\n",
        "    :param terminal_on_life_loss: If True, then step() returns done=True whenever a life is lost.\n",
        "    :param clip_reward: If True (default), the reward is clip to {-1, 0, 1} depending on its sign.\n",
        "    :param action_repeat_probability: Probability of repeating the last action\n",
        "    \"\"\"\n",
        "\n",
        "    def __init__(\n",
        "        self,\n",
        "        env: gym.Env,\n",
        "        noop_max: int = 30,\n",
        "        frame_skip: int = 4,\n",
        "        screen_size: int = 84,\n",
        "        terminal_on_life_loss: bool = True,\n",
        "        clip_reward: bool = True,\n",
        "        action_repeat_probability: float = 0.0,\n",
        "        normalize: bool = True,\n",
        "        video=None,\n",
        "    ) -> None:\n",
        "        if action_repeat_probability > 0.0:\n",
        "            env = StickyActionEnv(env, action_repeat_probability)\n",
        "        if noop_max > 0:\n",
        "            env = NoopResetEnv(env, noop_max=noop_max)\n",
        "        # frame_skip=1 is the same as no frame-skip (action repeat)\n",
        "        if frame_skip > 1:\n",
        "            env = MaxAndSkipEnv(env, skip=frame_skip)\n",
        "        if terminal_on_life_loss:\n",
        "            env = EpisodicLifeEnv(env)\n",
        "        if \"FIRE\" in env.unwrapped.get_action_meanings():  # type: ignore[attr-defined]\n",
        "            env = FireResetEnv(env)\n",
        "        env = WarpFrame(\n",
        "            env, width=screen_size, height=screen_size, normalize=normalize, video=video\n",
        "        )\n",
        "        if clip_reward:\n",
        "            env = ClipRewardEnv(env)\n",
        "\n",
        "        super().__init__(env)\n"
      ],
      "metadata": {
        "id": "3gMmyu_NJpFW"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 16,
      "metadata": {
        "id": "X7Ht-KCuIBu5",
        "outputId": "c657a5ae-2235-43a4-cebe-2ed443c3ac0f",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "device(type='cuda')"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ],
      "source": [
        "import gymnasium as gym\n",
        "import math\n",
        "import random\n",
        "import matplotlib\n",
        "import matplotlib.pyplot as plt\n",
        "from collections import namedtuple, deque\n",
        "from itertools import count\n",
        "import ale_py\n",
        "\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "import torch.nn.functional as F\n",
        "\n",
        "gym.register_envs(ale_py)\n",
        "\n",
        "env = gym.make(\"SpaceInvadersNoFrameskip-v4\")\n",
        "env = AtariWrapper(env)\n",
        "\n",
        "# set up matplotlib\n",
        "is_ipython = \"inline\" in matplotlib.get_backend()\n",
        "if is_ipython:\n",
        "    from IPython import display\n",
        "\n",
        "plt.ion()\n",
        "\n",
        "# if GPU is to be used\n",
        "device = torch.device(\n",
        "    \"cuda\"\n",
        "    if torch.cuda.is_available()\n",
        "    else \"mps\" if torch.backends.mps.is_available() else \"cpu\"\n",
        ")\n",
        "\n",
        "device"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "7d6J4ZdEIBu6"
      },
      "source": [
        "## Testing Env"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pip install -U colabgymrender"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "GBllu0O2LpUj",
        "outputId": "1820c492-b2d6-40a8-cb85-4adb1cbf7af0"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting colabgymrender\n",
            "  Downloading colabgymrender-1.1.0.tar.gz (3.5 kB)\n",
            "  Preparing metadata (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "Requirement already satisfied: moviepy in /usr/local/lib/python3.10/dist-packages (from colabgymrender) (1.0.3)\n",
            "Requirement already satisfied: decorator<5.0,>=4.0.2 in /usr/local/lib/python3.10/dist-packages (from moviepy->colabgymrender) (4.4.2)\n",
            "Requirement already satisfied: imageio<3.0,>=2.5 in /usr/local/lib/python3.10/dist-packages (from moviepy->colabgymrender) (2.36.1)\n",
            "Requirement already satisfied: imageio_ffmpeg>=0.2.0 in /usr/local/lib/python3.10/dist-packages (from moviepy->colabgymrender) (0.5.1)\n",
            "Requirement already satisfied: tqdm<5.0,>=4.11.2 in /usr/local/lib/python3.10/dist-packages (from moviepy->colabgymrender) (4.66.6)\n",
            "Requirement already satisfied: numpy>=1.17.3 in /usr/local/lib/python3.10/dist-packages (from moviepy->colabgymrender) (1.26.4)\n",
            "Requirement already satisfied: requests<3.0,>=2.8.1 in /usr/local/lib/python3.10/dist-packages (from moviepy->colabgymrender) (2.32.3)\n",
            "Requirement already satisfied: proglog<=1.0.0 in /usr/local/lib/python3.10/dist-packages (from moviepy->colabgymrender) (0.1.10)\n",
            "Requirement already satisfied: pillow>=8.3.2 in /usr/local/lib/python3.10/dist-packages (from imageio<3.0,>=2.5->moviepy->colabgymrender) (11.0.0)\n",
            "Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from imageio_ffmpeg>=0.2.0->moviepy->colabgymrender) (75.1.0)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3.0,>=2.8.1->moviepy->colabgymrender) (3.4.0)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3.0,>=2.8.1->moviepy->colabgymrender) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3.0,>=2.8.1->moviepy->colabgymrender) (2.2.3)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3.0,>=2.8.1->moviepy->colabgymrender) (2024.8.30)\n",
            "Building wheels for collected packages: colabgymrender\n",
            "  Building wheel for colabgymrender (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Created wheel for colabgymrender: filename=colabgymrender-1.1.0-py3-none-any.whl size=3114 sha256=56f266bd7bd0452f5700ffb916668719d0c1a33bf46b3a55dd4df4ff30384649\n",
            "  Stored in directory: /root/.cache/pip/wheels/13/62/63/7b3acfb684dd3d665d7fc1d213427b136205a222389767e295\n",
            "Successfully built colabgymrender\n",
            "Installing collected packages: colabgymrender\n",
            "Successfully installed colabgymrender-1.1.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 20,
      "metadata": {
        "id": "JETQP3ZbIBu6",
        "outputId": "cc96d205-be80-4f1e-b305-244397d95e21",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 422
        }
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "WARNING:py.warnings:/usr/local/lib/python3.10/dist-packages/gymnasium/envs/registration.py:517: DeprecationWarning: \u001b[33mWARN: The environment Breakout-v0 is out of date. You should consider upgrading to version `v4`.\u001b[0m\n",
            "  logger.deprecation(\n",
            "\n"
          ]
        },
        {
          "output_type": "error",
          "ename": "TypeError",
          "evalue": "OrderEnforcing.render() got an unexpected keyword argument 'mode'",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-20-ae0c3894723d>\u001b[0m in \u001b[0;36m<cell line: 29>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     27\u001b[0m \u001b[0menv_test\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mgym\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmake\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Breakout-v0\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     28\u001b[0m \u001b[0mdirectory\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m'./video'\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 29\u001b[0;31m \u001b[0menv_test\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mRecorder\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0menv_test\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdirectory\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     30\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     31\u001b[0m \u001b[0mtest_env\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0menv_test\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;32mlambda\u001b[0m \u001b[0mobs\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0menv\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0maction_space\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msample\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnum_episodes\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/colabgymrender/recorder.py\u001b[0m in \u001b[0;36m__init__\u001b[0;34m(self, env, directory, auto_release, size, fps)\u001b[0m\n\u001b[1;32m     18\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0msize\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     19\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0menv\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mreset\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 20\u001b[0;31m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msize\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0menv\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrender\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmode\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m'rgb_array'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;36m2\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m-\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     21\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     22\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msize\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0msize\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mTypeError\u001b[0m: OrderEnforcing.render() got an unexpected keyword argument 'mode'"
          ]
        }
      ],
      "source": [
        "from colabgymrender.recorder import Recorder\n",
        "\n",
        "def test_env(env_test, take_action, num_episodes=5):\n",
        "    for i in range(num_episodes):\n",
        "        observation, info = env_test.reset(seed=42)\n",
        "        done = False\n",
        "        score = 0\n",
        "\n",
        "        while not done:\n",
        "            # get next action\n",
        "            action = take_action(observation)\n",
        "\n",
        "            # take a step in the env\n",
        "            observation, reward, terminated, truncated, info = env_test.step(action)\n",
        "\n",
        "            # Collect rewards\n",
        "            score += reward\n",
        "\n",
        "            # set if done\n",
        "            done = terminated or truncated\n",
        "\n",
        "        print(f\"Episode {i+1}: {score}\")\n",
        "\n",
        "    env.close()\n",
        "\n",
        "\n",
        "env_test = gym.make(\"ALE/SpaceInvaders-v5\")\n",
        "directory = './video'\n",
        "env_test = Recorder(env_test, directory)\n",
        "\n",
        "test_env(env_test, lambda obs: env.action_space.sample(), num_episodes=1)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pip install imageio==2.4.1"
      ],
      "metadata": {
        "id": "GBE62f9INQY5",
        "outputId": "9ee57432-d182-48e2-a19d-9a86a6fb2272",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 481
        }
      },
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting imageio==2.4.1\n",
            "  Downloading imageio-2.4.1.tar.gz (3.3 MB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m3.3/3.3 MB\u001b[0m \u001b[31m31.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h  Preparing metadata (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from imageio==2.4.1) (1.26.4)\n",
            "Requirement already satisfied: pillow in /usr/local/lib/python3.10/dist-packages (from imageio==2.4.1) (11.0.0)\n",
            "Building wheels for collected packages: imageio\n",
            "  Building wheel for imageio (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Created wheel for imageio: filename=imageio-2.4.1-py3-none-any.whl size=3303884 sha256=d8aa9731b47352b3e325fb103686d78727276dc89633c769c21b1f65ab889e10\n",
            "  Stored in directory: /root/.cache/pip/wheels/96/5d/ce/bdbdb04744dac03906336eb0d01ff1e222061d3419c55c55f9\n",
            "Successfully built imageio\n",
            "Installing collected packages: imageio\n",
            "  Attempting uninstall: imageio\n",
            "    Found existing installation: imageio 2.36.1\n",
            "    Uninstalling imageio-2.36.1:\n",
            "      Successfully uninstalled imageio-2.36.1\n",
            "\u001b[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\n",
            "moviepy 1.0.3 requires imageio<3.0,>=2.5; python_version >= \"3.4\", but you have imageio 2.4.1 which is incompatible.\n",
            "scikit-image 0.24.0 requires imageio>=2.33, but you have imageio 2.4.1 which is incompatible.\u001b[0m\u001b[31m\n",
            "\u001b[0mSuccessfully installed imageio-2.4.1\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "application/vnd.colab-display-data+json": {
              "pip_warning": {
                "packages": [
                  "imageio"
                ]
              },
              "id": "4c747dcd306649f1b01e97b3b9daf97e"
            }
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import gym\n",
        "from colabgymrender.recorder import Recorder\n",
        "\n",
        "# print(list(gym.envs.registry.keys()))\n",
        "\n",
        "\n",
        "env = gym.make(\"CartPole-v0\")\n",
        "directory = './video'\n",
        "env = Recorder(env, directory)\n",
        "\n",
        "observation = env.reset()\n",
        "terminal = False\n",
        "while not terminal:\n",
        "  action = env.action_space.sample()\n",
        "  observation, reward, terminal, info = env.step(action)\n",
        "\n",
        "env.play()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 838
        },
        "id": "caeuSmnWMITc",
        "outputId": "cd0804c0-7349-4dc3-fec8-407e17a6e9b1"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/imageio/core/util.py:20: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html\n",
            "  import pkg_resources\n",
            "/usr/local/lib/python3.10/dist-packages/pkg_resources/__init__.py:3154: DeprecationWarning: Deprecated call to `pkg_resources.declare_namespace('google')`.\n",
            "Implementing implicit namespace packages (as specified in PEP 420) is preferred to `pkg_resources.declare_namespace`. See https://setuptools.pypa.io/en/latest/references/keywords.html#keyword-namespace-packages\n",
            "  declare_namespace(pkg)\n",
            "/usr/local/lib/python3.10/dist-packages/pkg_resources/__init__.py:3154: DeprecationWarning: Deprecated call to `pkg_resources.declare_namespace('google.cloud')`.\n",
            "Implementing implicit namespace packages (as specified in PEP 420) is preferred to `pkg_resources.declare_namespace`. See https://setuptools.pypa.io/en/latest/references/keywords.html#keyword-namespace-packages\n",
            "  declare_namespace(pkg)\n",
            "/usr/local/lib/python3.10/dist-packages/pkg_resources/__init__.py:3154: DeprecationWarning: Deprecated call to `pkg_resources.declare_namespace('sphinxcontrib')`.\n",
            "Implementing implicit namespace packages (as specified in PEP 420) is preferred to `pkg_resources.declare_namespace`. See https://setuptools.pypa.io/en/latest/references/keywords.html#keyword-namespace-packages\n",
            "  declare_namespace(pkg)\n",
            "WARNING:py.warnings:/usr/local/lib/python3.10/dist-packages/moviepy/video/fx/painting.py:7: DeprecationWarning: Please import `sobel` from the `scipy.ndimage` namespace; the `scipy.ndimage.filters` namespace is deprecated and will be removed in SciPy 2.0.0.\n",
            "  from scipy.ndimage.filters import sobel\n",
            "\n",
            "WARNING:py.warnings:/usr/local/lib/python3.10/dist-packages/gym/envs/registration.py:593: UserWarning: \u001b[33mWARN: The environment CartPole-v0 is out of date. You should consider upgrading to version `v1`.\u001b[0m\n",
            "  logger.warn(\n",
            "\n",
            "WARNING:py.warnings:/usr/local/lib/python3.10/dist-packages/gym/core.py:317: DeprecationWarning: \u001b[33mWARN: Initializing wrapper in old step API which returns one bool instead of two. It is recommended to set `new_step_api=True` to use new step API. This will be the default behaviour in future.\u001b[0m\n",
            "  deprecation(\n",
            "\n",
            "WARNING:py.warnings:/usr/local/lib/python3.10/dist-packages/gym/wrappers/step_api_compatibility.py:39: DeprecationWarning: \u001b[33mWARN: Initializing environment in old step API which returns one bool instead of two. It is recommended to set `new_step_api=True` to use new step API. This will be the default behaviour in future.\u001b[0m\n",
            "  deprecation(\n",
            "\n",
            "WARNING:py.warnings:/usr/local/lib/python3.10/dist-packages/gym/core.py:43: DeprecationWarning: \u001b[33mWARN: The argument mode in render method is deprecated; use render_mode during environment initialization instead.\n",
            "See here for more information: https://www.gymlibrary.ml/content/api/\u001b[0m\n",
            "  deprecation(\n",
            "\n",
            "WARNING:py.warnings:/usr/local/lib/python3.10/dist-packages/gym/utils/passive_env_checker.py:241: DeprecationWarning: `np.bool8` is a deprecated alias for `np.bool_`.  (Deprecated NumPy 1.24)\n",
            "  if not isinstance(terminated, (bool, np.bool8)):\n",
            "\n"
          ]
        },
        {
          "output_type": "error",
          "ename": "TypeError",
          "evalue": "VideoClip.write_videofile() got an unexpected keyword argument 'progress_bar'",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-1-d089992bf465>\u001b[0m in \u001b[0;36m<cell line: 17>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     15\u001b[0m   \u001b[0mobservation\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mreward\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mterminal\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0minfo\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0menv\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstep\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0maction\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     16\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 17\u001b[0;31m \u001b[0menv\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mplay\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/colabgymrender/recorder.py\u001b[0m in \u001b[0;36mplay\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m     70\u001b[0m         \u001b[0mfilename\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m'temp-{start}.mp4'\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     71\u001b[0m         \u001b[0mclip\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mVideoFileClip\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpath\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 72\u001b[0;31m         \u001b[0mclip\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mwrite_videofile\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfilename\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mprogress_bar\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mFalse\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mverbose\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mFalse\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     73\u001b[0m         \u001b[0mdisplay\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mVideo\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfilename\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0membed\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     74\u001b[0m         \u001b[0mos\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mremove\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfilename\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mTypeError\u001b[0m: VideoClip.write_videofile() got an unexpected keyword argument 'progress_bar'"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "dk4BgYISIBu6"
      },
      "source": [
        "## DQN"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "7_A2XvxuIBu6"
      },
      "outputs": [],
      "source": [
        "class DQN(nn.Module):\n",
        "    def __init__(self, in_channels, n_actions):\n",
        "        super().__init__()\n",
        "        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)\n",
        "        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)\n",
        "        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)\n",
        "        self.fc1 = nn.Linear(64 * 7 * 7, 512)\n",
        "        self.fc2 = nn.Linear(512, n_actions)\n",
        "\n",
        "    def forward(self, x):\n",
        "        \"\"\"\n",
        "        Input shape: (bs,c,h,w) #(bs,4,84,84)\n",
        "        Output shape: (bs,n_actions)\n",
        "        \"\"\"\n",
        "        x = F.relu(self.conv1(x))\n",
        "        x = F.relu(self.conv2(x))\n",
        "        x = F.relu(self.conv3(x))\n",
        "        x = torch.flatten(x, 1)\n",
        "        x = F.relu(self.fc1(x))\n",
        "        return self.fc2(x)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "SuzsaLILIBu6"
      },
      "source": [
        "## Replay Memory"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ul_Thz31IBu7"
      },
      "outputs": [],
      "source": [
        "Transition = namedtuple(\"Transition\", (\"state\", \"action\", \"next_state\", \"reward\"))\n",
        "\n",
        "\n",
        "class ReplayMemory(object):\n",
        "\n",
        "    def __init__(self, capacity):\n",
        "        self.memory = deque([], maxlen=capacity)\n",
        "\n",
        "    def push(self, *args):\n",
        "        \"\"\"Save a transition\"\"\"\n",
        "        self.memory.append(Transition(*args))\n",
        "\n",
        "    def sample(self, batch_size):\n",
        "        return random.sample(self.memory, batch_size)\n",
        "\n",
        "    def __len__(self):\n",
        "        return len(self.memory)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "mkFw1Bz7IBu7"
      },
      "outputs": [],
      "source": [
        "# BATCH_SIZE is the number of transitions sampled from the replay buffer\n",
        "# GAMMA is the discount factor as mentioned in the previous section\n",
        "# EPS_START is the starting value of epsilon\n",
        "# EPS_END is the final value of epsilon\n",
        "# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay\n",
        "# TAU is the update rate of the target network\n",
        "# LR is the learning rate of the ``AdamW`` optimizer\n",
        "\n",
        "\n",
        "BATCH_SIZE = 128\n",
        "GAMMA = 0.99\n",
        "EPS_START = 0.9\n",
        "EPS_END = 0.05\n",
        "EPS_DECAY = 1000\n",
        "TAU = 0.005\n",
        "LR = 1e-4\n",
        "\n",
        "# Get number of actions from gym action space\n",
        "n_actions = env.action_space.n\n",
        "# Get the number of state observations\n",
        "state, info = env.reset()\n",
        "n_observations = len(state)\n",
        "\n",
        "policy_net = DQN(in_channels=4, n_actions=n_actions).to(device)\n",
        "target_net = DQN(in_channels=4, n_actions=n_actions).to(device)\n",
        "\n",
        "target_net.load_state_dict(policy_net.state_dict())\n",
        "target_net.eval()\n",
        "\n",
        "\n",
        "optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)\n",
        "memory = ReplayMemory(50000)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "EtQlI3O0IBu7"
      },
      "source": [
        "## Epsilon Greey"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "VPUtmHeUIBu7"
      },
      "outputs": [],
      "source": [
        "steps_done = 0\n",
        "\n",
        "\n",
        "def select_action(state: torch.Tensor) -> torch.Tensor:\n",
        "    \"\"\"\n",
        "    epsilon greedy\n",
        "    - epsilon: choose random action\n",
        "    - 1-epsilon: argmax Q(a,s)\n",
        "\n",
        "    Input: state shape (1,4,84,84)\n",
        "\n",
        "    Output: action shape (1,1)\n",
        "    \"\"\"\n",
        "    global eps_threshold\n",
        "    global steps_done\n",
        "    sample = random.random()\n",
        "    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(\n",
        "        -1.0 * steps_done / EPS_DECAY\n",
        "    )\n",
        "    steps_done += 1\n",
        "    if sample > eps_threshold:\n",
        "        with torch.no_grad():\n",
        "            return policy_net(state).max(1)[1].view(1, 1)\n",
        "    else:\n",
        "        return torch.tensor([[env.action_space.sample()]]).to(device)\n",
        "\n",
        "\n",
        "# def select_action(state):\n",
        "#     global steps_done\n",
        "#     sample = random.random()\n",
        "#     eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(\n",
        "#         -1.0 * steps_done / EPS_DECAY\n",
        "#     )\n",
        "#     steps_done += 1\n",
        "#     if sample > eps_threshold:\n",
        "#         with torch.no_grad():\n",
        "#             # t.max(1) will return the largest column value of each row.\n",
        "#             # second column on max result is index of where max element was\n",
        "#             # found, so we pick action with the larger expected reward.\n",
        "#             return policy_net(state).max(1).indices.view(1, 1)\n",
        "#     else:\n",
        "#         return torch.tensor(\n",
        "#             [[env.action_space.sample()]], device=device, dtype=torch.long\n",
        "#         )"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-d8naoGnIBu7"
      },
      "source": [
        "## Plot Duration"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "XMvV-nO7IBu7"
      },
      "outputs": [],
      "source": [
        "episode_durations = []\n",
        "\n",
        "\n",
        "def plot_durations(show_result=False):\n",
        "    plt.figure(1)\n",
        "    durations_t = torch.tensor(episode_durations, dtype=torch.float)\n",
        "    if show_result:\n",
        "        plt.title(\"Result\")\n",
        "    else:\n",
        "        plt.clf()\n",
        "        plt.title(\"Training...\")\n",
        "    plt.xlabel(\"Episode\")\n",
        "    plt.ylabel(\"Duration\")\n",
        "    plt.plot(durations_t.numpy())\n",
        "    # Take 100 episode averages and plot them too\n",
        "    if len(durations_t) >= 100:\n",
        "        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)\n",
        "        means = torch.cat((torch.zeros(99), means))\n",
        "        plt.plot(means.numpy())\n",
        "\n",
        "    plt.pause(0.001)  # pause a bit so that plots are updated\n",
        "    if is_ipython:\n",
        "        if not show_result:\n",
        "            display.display(plt.gcf())\n",
        "            display.clear_output(wait=True)\n",
        "        else:\n",
        "            display.display(plt.gcf())"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "rpSJC3wwIBu7"
      },
      "source": [
        "## Optimize"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0rQOPoViIBu7"
      },
      "outputs": [],
      "source": [
        "def optimize_model():\n",
        "    if len(memory) < BATCH_SIZE:\n",
        "        return\n",
        "    transitions = memory.sample(BATCH_SIZE)\n",
        "    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for\n",
        "    # detailed explanation). This converts batch-array of Transitions\n",
        "    # to Transition of batch-arrays.\n",
        "    batch = Transition(*zip(*transitions))\n",
        "\n",
        "    # Compute a mask of non-final states and concatenate the batch elements\n",
        "    # (a final state would've been the one after which simulation ended)\n",
        "    non_final_mask = torch.tensor(\n",
        "        tuple(map(lambda s: s is not None, batch.next_state)),\n",
        "        device=device,\n",
        "        dtype=torch.bool,\n",
        "    )\n",
        "    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])\n",
        "    state_batch = torch.cat(batch.state)\n",
        "    action_batch = torch.cat(batch.action)\n",
        "    reward_batch = torch.cat(batch.reward)\n",
        "\n",
        "    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the\n",
        "    # columns of actions taken. These are the actions which would've been taken\n",
        "    # for each batch state according to policy_net\n",
        "    state_action_values = policy_net(state_batch).gather(1, action_batch)\n",
        "\n",
        "    # Compute V(s_{t+1}) for all next states.\n",
        "    # Expected values of actions for non_final_next_states are computed based\n",
        "    # on the \"older\" target_net; selecting their best reward with max(1).values\n",
        "    # This is merged based on the mask, such that we'll have either the expected\n",
        "    # state value or 0 in case the state was final.\n",
        "    next_state_values = torch.zeros(BATCH_SIZE, device=device)\n",
        "    with torch.no_grad():\n",
        "        next_state_values[non_final_mask] = (\n",
        "            target_net(non_final_next_states).max(1).values\n",
        "        )\n",
        "    # Compute the expected Q values\n",
        "    expected_state_action_values = (next_state_values * GAMMA) + reward_batch\n",
        "\n",
        "    # Compute Huber loss\n",
        "    criterion = nn.SmoothL1Loss()\n",
        "    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))\n",
        "\n",
        "    # Optimize the model\n",
        "    optimizer.zero_grad()\n",
        "    loss.backward()\n",
        "    # In-place gradient clipping\n",
        "    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)\n",
        "    optimizer.step()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "3J_oNp5oIBu8"
      },
      "source": [
        "## Train loop"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "NdB8dZYvIBu8",
        "outputId": "9dc9ac70-372a-40f9-b299-9ed7df73c4f9"
      },
      "outputs": [
        {
          "ename": "NameError",
          "evalue": "name 'torch' is not defined",
          "output_type": "error",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "Cell \u001b[0;32mIn[1], line 1\u001b[0m\n\u001b[0;32m----> 1\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[43mtorch\u001b[49m\u001b[38;5;241m.\u001b[39mcuda\u001b[38;5;241m.\u001b[39mis_available() \u001b[38;5;129;01mor\u001b[39;00m torch\u001b[38;5;241m.\u001b[39mbackends\u001b[38;5;241m.\u001b[39mmps\u001b[38;5;241m.\u001b[39mis_available():\n\u001b[1;32m      2\u001b[0m     num_episodes \u001b[38;5;241m=\u001b[39m \u001b[38;5;241m600\u001b[39m\n\u001b[1;32m      3\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n",
            "\u001b[0;31mNameError\u001b[0m: name 'torch' is not defined"
          ]
        }
      ],
      "source": [
        "if torch.cuda.is_available() or torch.backends.mps.is_available():\n",
        "    num_episodes = 600\n",
        "else:\n",
        "    num_episodes = 50\n",
        "\n",
        "for i_episode in range(num_episodes):\n",
        "    # Initialize the environment and get its state\n",
        "    state, info = env.reset()\n",
        "\n",
        "    state = torch.from_numpy(state).to(device)  # (84,84)\n",
        "    # stack four frames together, hoping to learn temporal info\n",
        "    state = torch.stack((state, state, state, state)).unsqueeze(0)  # (1,4,84,84)\n",
        "\n",
        "    for t in count():\n",
        "        action = select_action(state)\n",
        "        obs, reward, terminated, truncated, _ = env.step(action.item())\n",
        "        reward = torch.tensor([reward], device=device)\n",
        "        done = terminated or truncated\n",
        "\n",
        "        if terminated:\n",
        "            next_state = None\n",
        "        else:\n",
        "            next_state = torch.from_numpy(obs).to(device)  # (84,84)\n",
        "            next_state = torch.stack(\n",
        "                (next_state, obs[0][0], obs[0][1], obs[0][2])\n",
        "            ).unsqueeze(\n",
        "                0\n",
        "            )  # (1,4,84,84)\n",
        "            # next_state = torch.tensor(\n",
        "            #     observation, dtype=torch.float32, device=device\n",
        "            # ).unsqueeze(0)\n",
        "\n",
        "        # Store the transition in memory\n",
        "        memory.push(state, action, next_state, reward)\n",
        "\n",
        "        # Move to the next state\n",
        "        state = next_state\n",
        "\n",
        "        # Perform one step of the optimization (on the policy network)\n",
        "        optimize_model()\n",
        "\n",
        "        # Soft update of the target network's weights\n",
        "        # θ′ ← τ θ + (1 −τ )θ′\n",
        "        target_net_state_dict = target_net.state_dict()\n",
        "        policy_net_state_dict = policy_net.state_dict()\n",
        "        for key in policy_net_state_dict:\n",
        "            target_net_state_dict[key] = policy_net_state_dict[\n",
        "                key\n",
        "            ] * TAU + target_net_state_dict[key] * (1 - TAU)\n",
        "        target_net.load_state_dict(target_net_state_dict)\n",
        "\n",
        "        if done:\n",
        "            episode_durations.append(t + 1)\n",
        "            plot_durations()\n",
        "            break\n",
        "\n",
        "print(\"Complete\")\n",
        "plot_durations(show_result=True)\n",
        "plt.ioff()\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "6i3wn1sFIBu8"
      },
      "source": [
        "## Test agent"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "e5VCDFY-IBu8",
        "outputId": "92cb288e-7520-4e3c-995e-19baec8ed716"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Episode 1: 500.0\n"
          ]
        },
        {
          "ename": "KeyboardInterrupt",
          "evalue": "",
          "output_type": "error",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
            "Cell \u001b[0;32mIn[22], line 23\u001b[0m\n\u001b[1;32m     19\u001b[0m     action \u001b[38;5;241m=\u001b[39m policy_net(observation)\u001b[38;5;241m.\u001b[39mmax(\u001b[38;5;241m1\u001b[39m)\u001b[38;5;241m.\u001b[39mindices\u001b[38;5;241m.\u001b[39mview(\u001b[38;5;241m1\u001b[39m, \u001b[38;5;241m1\u001b[39m)\n\u001b[1;32m     21\u001b[0m \u001b[38;5;66;03m# step (transition) through the environment with the action\u001b[39;00m\n\u001b[1;32m     22\u001b[0m \u001b[38;5;66;03m# receiving the next observation, reward and if the episode has terminated or truncated\u001b[39;00m\n\u001b[0;32m---> 23\u001b[0m observation, reward, terminated, truncated, info \u001b[38;5;241m=\u001b[39m \u001b[43menv\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mstep\u001b[49m\u001b[43m(\u001b[49m\u001b[43maction\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mitem\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m     25\u001b[0m score \u001b[38;5;241m+\u001b[39m\u001b[38;5;241m=\u001b[39m reward\n\u001b[1;32m     26\u001b[0m \u001b[38;5;66;03m# set if done\u001b[39;00m\n",
            "File \u001b[0;32m~/Git/pytorch-demo/.venv/lib/python3.10/site-packages/gymnasium/wrappers/common.py:125\u001b[0m, in \u001b[0;36mTimeLimit.step\u001b[0;34m(self, action)\u001b[0m\n\u001b[1;32m    112\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mstep\u001b[39m(\n\u001b[1;32m    113\u001b[0m     \u001b[38;5;28mself\u001b[39m, action: ActType\n\u001b[1;32m    114\u001b[0m ) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m \u001b[38;5;28mtuple\u001b[39m[ObsType, SupportsFloat, \u001b[38;5;28mbool\u001b[39m, \u001b[38;5;28mbool\u001b[39m, \u001b[38;5;28mdict\u001b[39m[\u001b[38;5;28mstr\u001b[39m, Any]]:\n\u001b[1;32m    115\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"Steps through the environment and if the number of steps elapsed exceeds ``max_episode_steps`` then truncate.\u001b[39;00m\n\u001b[1;32m    116\u001b[0m \n\u001b[1;32m    117\u001b[0m \u001b[38;5;124;03m    Args:\u001b[39;00m\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m    123\u001b[0m \n\u001b[1;32m    124\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[0;32m--> 125\u001b[0m     observation, reward, terminated, truncated, info \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43menv\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mstep\u001b[49m\u001b[43m(\u001b[49m\u001b[43maction\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    126\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_elapsed_steps \u001b[38;5;241m+\u001b[39m\u001b[38;5;241m=\u001b[39m \u001b[38;5;241m1\u001b[39m\n\u001b[1;32m    128\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_elapsed_steps \u001b[38;5;241m>\u001b[39m\u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_max_episode_steps:\n",
            "File \u001b[0;32m~/Git/pytorch-demo/.venv/lib/python3.10/site-packages/gymnasium/wrappers/common.py:393\u001b[0m, in \u001b[0;36mOrderEnforcing.step\u001b[0;34m(self, action)\u001b[0m\n\u001b[1;32m    391\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_has_reset:\n\u001b[1;32m    392\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m ResetNeeded(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mCannot call env.step() before calling env.reset()\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m--> 393\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43msuper\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mstep\u001b[49m\u001b[43m(\u001b[49m\u001b[43maction\u001b[49m\u001b[43m)\u001b[49m\n",
            "File \u001b[0;32m~/Git/pytorch-demo/.venv/lib/python3.10/site-packages/gymnasium/core.py:322\u001b[0m, in \u001b[0;36mWrapper.step\u001b[0;34m(self, action)\u001b[0m\n\u001b[1;32m    318\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mstep\u001b[39m(\n\u001b[1;32m    319\u001b[0m     \u001b[38;5;28mself\u001b[39m, action: WrapperActType\n\u001b[1;32m    320\u001b[0m ) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m \u001b[38;5;28mtuple\u001b[39m[WrapperObsType, SupportsFloat, \u001b[38;5;28mbool\u001b[39m, \u001b[38;5;28mbool\u001b[39m, \u001b[38;5;28mdict\u001b[39m[\u001b[38;5;28mstr\u001b[39m, Any]]:\n\u001b[1;32m    321\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"Uses the :meth:`step` of the :attr:`env` that can be overwritten to change the returned data.\"\"\"\u001b[39;00m\n\u001b[0;32m--> 322\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43menv\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mstep\u001b[49m\u001b[43m(\u001b[49m\u001b[43maction\u001b[49m\u001b[43m)\u001b[49m\n",
            "File \u001b[0;32m~/Git/pytorch-demo/.venv/lib/python3.10/site-packages/gymnasium/wrappers/common.py:285\u001b[0m, in \u001b[0;36mPassiveEnvChecker.step\u001b[0;34m(self, action)\u001b[0m\n\u001b[1;32m    283\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m env_step_passive_checker(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39menv, action)\n\u001b[1;32m    284\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m--> 285\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43menv\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mstep\u001b[49m\u001b[43m(\u001b[49m\u001b[43maction\u001b[49m\u001b[43m)\u001b[49m\n",
            "File \u001b[0;32m~/Git/pytorch-demo/.venv/lib/python3.10/site-packages/gymnasium/envs/classic_control/cartpole.py:223\u001b[0m, in \u001b[0;36mCartPoleEnv.step\u001b[0;34m(self, action)\u001b[0m\n\u001b[1;32m    220\u001b[0m     reward \u001b[38;5;241m=\u001b[39m \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m1.0\u001b[39m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_sutton_barto_reward \u001b[38;5;28;01melse\u001b[39;00m \u001b[38;5;241m0.0\u001b[39m\n\u001b[1;32m    222\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mrender_mode \u001b[38;5;241m==\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mhuman\u001b[39m\u001b[38;5;124m\"\u001b[39m:\n\u001b[0;32m--> 223\u001b[0m     \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mrender\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    225\u001b[0m \u001b[38;5;66;03m# truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`\u001b[39;00m\n\u001b[1;32m    226\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m np\u001b[38;5;241m.\u001b[39marray(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mstate, dtype\u001b[38;5;241m=\u001b[39mnp\u001b[38;5;241m.\u001b[39mfloat32), reward, terminated, \u001b[38;5;28;01mFalse\u001b[39;00m, {}\n",
            "File \u001b[0;32m~/Git/pytorch-demo/.venv/lib/python3.10/site-packages/gymnasium/envs/classic_control/cartpole.py:337\u001b[0m, in \u001b[0;36mCartPoleEnv.render\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    335\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mrender_mode \u001b[38;5;241m==\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mhuman\u001b[39m\u001b[38;5;124m\"\u001b[39m:\n\u001b[1;32m    336\u001b[0m     pygame\u001b[38;5;241m.\u001b[39mevent\u001b[38;5;241m.\u001b[39mpump()\n\u001b[0;32m--> 337\u001b[0m     \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mclock\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mtick\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mmetadata\u001b[49m\u001b[43m[\u001b[49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mrender_fps\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m]\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    338\u001b[0m     pygame\u001b[38;5;241m.\u001b[39mdisplay\u001b[38;5;241m.\u001b[39mflip()\n\u001b[1;32m    340\u001b[0m \u001b[38;5;28;01melif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mrender_mode \u001b[38;5;241m==\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mrgb_array\u001b[39m\u001b[38;5;124m\"\u001b[39m:\n",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
          ]
        }
      ],
      "source": [
        "import gymnasium as gym\n",
        "\n",
        "# Initialise the environment\n",
        "env = gym.make(\"CartPole-v1\", render_mode=\"human\")\n",
        "\n",
        "# Run 5 episodes\n",
        "num_episodes = 5\n",
        "\n",
        "for i in range(num_episodes):\n",
        "    observation, info = env.reset(seed=42)\n",
        "    done = False\n",
        "    score = 0\n",
        "\n",
        "    while not done:\n",
        "        with torch.no_grad():\n",
        "            observation = torch.tensor(\n",
        "                observation, dtype=torch.float32, device=device\n",
        "            ).unsqueeze(0)\n",
        "            action = policy_net(observation).max(1).indices.view(1, 1)\n",
        "\n",
        "        # step (transition) through the environment with the action\n",
        "        # receiving the next observation, reward and if the episode has terminated or truncated\n",
        "        observation, reward, terminated, truncated, info = env.step(action.item())\n",
        "\n",
        "        # Collect rewards\n",
        "        score += reward\n",
        "\n",
        "        # set if done\n",
        "        done = terminated or truncated\n",
        "\n",
        "    print(f\"Episode {i+1}: {score}\")\n",
        "\n",
        "env.close()"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.10.12"
    },
    "colab": {
      "provenance": [],
      "machine_shape": "hm",
      "gpuType": "A100",
      "include_colab_link": true
    },
    "accelerator": "GPU"
  },
  "nbformat": 4,
  "nbformat_minor": 0
}