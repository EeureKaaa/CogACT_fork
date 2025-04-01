from PIL import Image
import torch
import logging
import asyncio
import json
import numpy as np
import websockets
from openpi_client import msgpack_numpy
from sim_cogact.cogact_policy import CogACTInference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load CogACT inference model
def load_model():
    logger.info("Loading CogACT inference model...")
    model = CogACTInference(
        saved_model_path='/home/wangxianhao/data/project/reasoning/CogACT/logdir/checkpoint_dir/cogact_libero_finetune_v1--image_aug/checkpoints/step-057000-epoch-71-loss=0.0083.pt',  # choose from [CogACT-Small, CogACT-Base, CogACT-Large] or the local path
        action_model_type='DiT-B',              # choose from ['DiT-S', 'DiT-B', 'DiT-L'] to match the model weight
        future_action_window_size=15,
        unnorm_key='custom_finetuning',      # input your unnorm_key of the dataset
        cfg_scale=3,                            # cfg from 1.5 to 7 also performs well
        use_ddim=True,                          # use DDIM sampling
        num_ddim_steps=10,                      # number of steps for DDIM sampling
        action_scale=1.0,                       # scale factor for actions
        action_ensemble=True,                   # enable action ensembling
        adaptive_ensemble_alpha=0.1,            # weight parameter for adaptive ensembling
        policy_setup="google_robot",            # policy setup for CogACT
    )
    logger.info("CogACT inference model loaded successfully")
    return model

# Process observation and prompt to generate actions
async def process_observation(model, observation, prompt):
    
    # Extract image from observation
    if 'image' in observation:
        # Convert numpy array to PIL Image if needed
        image_array = observation['image']
        
        # Set the task description for the model
        model.task_description = prompt
        
        # Step the model with the image to get actions
        action_dict, _ = model.step(image_array)
        
        # Extract the raw actions from the raw action dictionary
        world_vector = action_dict["world_vector"]
        rotation_delta = action_dict["rotation_delta"]  
        open_gripper = action_dict["open_gripper"]
        
        # Combine into a single action vector (7-dimensional)
        action = np.concatenate([
            world_vector,
            rotation_delta,
            open_gripper
        ])
        
        # Reshape to match expected output format [1, 7]
        actions_np = np.expand_dims(action, axis=0)
        
        logger.info(f"Generated action: {action}")
    else:
        raise ValueError("Observation does not contain an image")

    return {
        'actions': actions_np,
        'status': 'success'
    }

# WebSocket server handler
async def websocket_handler(websocket):
    # Send metadata to client
    metadata = {
        'server_name': 'CogACT Action Server',
        'version': '1.0',
        'capabilities': ['action_generation']
    }
    await websocket.send(msgpack_numpy.packb(metadata))
    
    logger.info("Client connected. Metadata sent.")
    
    try:
        async for message in websocket:
            try:
                # Unpack the message
                data = msgpack_numpy.unpackb(message)
                logger.info(f"Received data with keys: {data.keys()}")
                
                # Extract image and prompt directly from the data
                image = data.get('image')
                prompt = data.get('prompt', '')
                
                if not prompt:
                    logger.warning("Received empty prompt, using default")
                    prompt = "pick up the object"
                
                logger.info(f"Processing with prompt: '{prompt}' and image shape: {image.shape if image is not None else 'None'}")
                
                # Create observation with the image for process_observation
                observation = {'image': image}
                
                # Process the observation and generate actions
                result = await process_observation(model, observation, prompt)
                
                # Send back the result
                await websocket.send(msgpack_numpy.packb(result))
                logger.info("Sent action response to client")
                
            except Exception as e:
                error_msg = f"Error processing message: {str(e)}"
                logger.error(error_msg)
                await websocket.send(msgpack_numpy.packb({'error': error_msg, 'status': 'error'}))
    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")

# Start the WebSocket server
async def start_server(host='0.0.0.0', port=8000):
    server = await websockets.serve(
        websocket_handler, 
        host, 
        port,
        max_size=None,  # No limit on message size
        compression=None  # Disable compression for performance
    )
    
    logger.info(f"WebSocket server started at ws://{host}:{port}")
    
    return server

# Main function
async def main():
    global model
    
    # Load the CogACT model
    model = load_model()
    
    # Start the WebSocket server
    server = await start_server()
    
    # Keep the server running
    await asyncio.Future()

# Run the server
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")