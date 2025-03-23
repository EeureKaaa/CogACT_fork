# Policy Finetuning and ManiSkill Integration Guide

This guide outlines the general methodology for finetuning robotic policies and integrating them with ManiSkill.

## Table of Contents
- [Policy Finetuning Pipeline](#policy-finetuning-pipeline)
- [ManiSkill Integration via WebSocket](#maniskill-integration-via-websocket)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Policy Finetuning Pipeline

### 1. Dataset Preparation

Regardless of the policy architecture, the dataset preparation follows similar principles:

1. **Dataset Format**: 
   - Use a standardized format like RLDS (Robot Learning Dataset Schema)(e.g. CogACT) or LeRobot (e.g. pi0)
   - Ensure consistent observation and action spaces
      for CogACT, obs: external-camera image; action: 7-dim actions [∆x, ∆y, ∆z, ∆φ, ∆θ, ∆ψ, g]; instruction: text instruction

   - Include task descriptions or instructions if relevant

2. **Data Structure**:
   - for RLDS datasets, the structure is like:
   ```
   └── custom_dataset/
    └── 1.0.0/
        ├── dataset_info.json
        ├── features.json
        ├── metadata.json
        ├── train-00000-of-00010.tfrecord
        ├── train-00001-of-00010.tfrecord
        └── ... (more tfrecord files)
   ```
   - you can also use [rlds dataset builder](https://github.com/kpertsch/rlds_dataset_builder) to build your dataset
   - LIBERO dataset: https://huggingface.co/datasets/openvla/modified_libero_rlds

   - for LeRobot datasets, the structure is like:
   - you can use convert_libero_data script(by Physical Intelligence) to convert your dataset: https://github.com/Physical-Intelligence/openpi


### 2. Finetuning Process

The finetuning process typically involves:

1. **Loading Base Model**:
   - Start with a pre-trained model
   - Configure model parameters based on your robot's requirements
   - for gated models, make sure to request official access, and generate an access token
   - '''export HF_TOKEN=your_token'''

2. **Computing Dataset Statistics(optional)**:
   - Calculate action statistics (mean, std) for normalization
   - Create a unique identifier (e.g., `unnorm_key`) for these statistics
   - check if you need to adjust scaling parameters

3. **Training Configuration**:
   - Set hyperparameters (learning rate, batch size, etc.), and just follow the instructions in the finetuning script

4. **Model Selection**:
   - Choose the best checkpoint based on validation performance
   - Consider metrics relevant to your robotic tasks

## ManiSkill Integration via WebSocket

### 1. Server-Client Architecture

The WebSocket-based server-client architecture enables separation of policy inference and environment simulation
To install dependencies, you need to clone openpi_client:
```bash
git clone https://github.com/Physical-Intelligence/openpi.git
```
then jump into packages/openpi-client and install dependencies in your virtual environment:
```bash
pip install -e .
```
### 2. Server Implementation

1. **Model Loading**:

To start the policy server, you can take test_connect.py as an example

   - you need to load the finetuned model in the load_model function
   ```python
   def load_model():
       model = PolicyClass(
           saved_model_path='/path/to/checkpoint.pt',
           # Model-specific parameters
           # ...
           # Environment-specific parameters
           # ...
       )
       return model
   ```

2. **Observation Processing**:
   - you need to adjust the process_observation function to match your policy model process
   ```python
   async def process_observation(model, observation, instruction):
       # Extract relevant information from observation
       # Process through model to get actions
       # Format actions for the environment
       return {
           'actions': actions,
           'status': 'success'
       }
   ```

3. **WebSocket Handler**:
   ```python
   async def websocket_handler(websocket):
       # Initialize model
       # Handle incoming connections
       # Process messages
       # Send responses
   ```

The given host is "0.0.0.0" and the port is "8000"
To start the server, you can run:
```bash
python test_connect.py
```

### 3. Client Implementation

   - To implement the client policy, you need to add your policy to connect_test_universal_tabletop.py
   - Adjust the client massage depending on your policy, for CogACT, it should be like:
   ```python
   cogact_obs = {
       'image': image,
       'prompt': args.prompt
   }
   ```
   - Then send and receive the message
   ```python
   result = client.infer(cogact_obs)
   ```
   - You can adjust action mapping scale by setting scale parameter in the mapping function
   ```python
   def mapping_function(action, scale):
       # Your mapping logic here
       return scaled_action
   ```
If you are in headless mode, you need to run:
```bash
xvfb-run -a python env_tests/connect_test_universal_tabletop.py --use-cogact --cogact-host localhost --cogact-port 8000 --prompt "pick up the apple" --save-images --use-external-camera
```
If not, you can run:
```bash
python env_tests/connect_test_universal_tabletop.py --use-cogact --cogact-host localhost --cogact-port 8000 --prompt "pick up the apple" --save-images --use-external-camera
```
and modify the parameters in the script to match your policy

## Best Practices

When adapting the pipeline for different policy architectures:

1. **Action Space**:
   - Ensure the action space matches the environment's expectations
   - You can change 'control_mode' to change the action space in the script to match your policy
   - Apply appropriate scaling and normalization

2. **Observation Processing**:
   - Handle different observation modalities (images, states, etc.)
   - Preprocess observations to match model's expected input format, external-camera image or hand-camera image...

3. **Robot-Specific Parameters**:
   - Adjust policy parameters based on the robot type according to your policy model setup


By following this guide, you can finetune your policy models and integrate them with ManiSkill or similar robot simulation environments.