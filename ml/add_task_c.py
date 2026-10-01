import json

def append_to_notebook():
    file_path = 'd:\\Projects\\F1-Strategy-Predictor-main\\ml\\f1_strategy_rl.ipynb'
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    md_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Bagian C & D: RL Environment & Training\n",
            "\n",
            "Sesuai dengan instruksi tugas, kita menggunakan `F1StrategyEnv` sebagai model-based simulator untuk melatih agen PPO atau DQN."
        ]
    }
    
    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from stable_baselines3 import PPO\n",
            "from stable_baselines3.common.env_checker import check_env\n",
            "import lightgbm as lgb\n",
            "import joblib\n",
            "\n",
            "# 1. Pastikan model Bagian B sudah dimuat\n",
            "# gbm = joblib.load(f\"{MODEL_DIR}/lap_time_lgbm.joblib\")\n",
            "\n",
            "# 2. Inisialisasi Environment\n",
            "env = F1StrategyEnv(\n",
            "    predictor_model=gbm,\n",
            "    race_laps=50,\n",
            "    circuit=\"Abu Dhabi Grand Prix\",\n",
            "    team=\"Mercedes\",\n",
            "    initial_rolling_pace=90.0\n",
            ")\n",
            "\n",
            "# Validasi environment menggunakan utility dari Stable-Baselines3\n",
            "check_env(env, warn=True)\n",
            "\n",
            "# 3. Training Agent menggunakan PPO\n",
            "print(\"Memulai training PPO...\")\n",
            "model_ppo = PPO(\"MlpPolicy\", env, verbose=1, learning_rate=0.001)\n",
            "model_ppo.learn(total_timesteps=10000)\n",
            "print(\"Training selesai!\")\n",
            "\n",
            "# 4. Evaluasi (Contoh satu race)\n",
            "obs, info = env.reset()\n",
            "done = False\n",
            "total_reward = 0\n",
            "\n",
            "print(\"\\n--- Hasil Simulasi Satu Race ---\")\n",
            "while not done:\n",
            "    action, _states = model_ppo.predict(obs, deterministic=True)\n",
            "    obs, reward, terminated, truncated, info = env.step(action)\n",
            "    total_reward += reward\n",
            "    done = terminated or truncated\n",
            "    if action != 0:\n",
            "        print(f\"Lap {50 - obs[0]}: PIT STOP ke compound index {action - 1}\")\n",
            "\n",
            "print(f\"Total Race Time (negated reward): {-total_reward:.2f} detik\")\n"
        ]
    }
    
    nb['cells'].append(md_cell)
    nb['cells'].append(code_cell)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
if __name__ == '__main__':
    append_to_notebook()
