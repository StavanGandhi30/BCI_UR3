# BCI\_UR3

## Setting Up the Environment

1. **Create a Conda environment**

```bash
conda create -n bci python=3.x
```

2. **Activate the environment**

```bash
conda activate bci
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Create a `.env` file** in the project root and add your credentials:

```
client_id=your_client_id
client_secret=your_client_secret
profile_name=your_profile_name
```

> Replace `your_client_id`, `your_client_secret`, and `your_profile_name` with the actual values.
