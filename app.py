import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained DQN model
model = load_model("dqn_cartpole.h5", compile=False)

# Define a prediction function
def predict_action(pos, vel, angle, ang_vel):
    try:
        state = np.array([[float(pos), float(vel), float(angle), float(ang_vel)]])
        q_values = model.predict(state, verbose=0)
        action = int(np.argmax(q_values[0]))
        return str(action), q_values[0].tolist()
    except Exception as e:
        return f"Error: {str(e)}", "Error"

# Gradio interface
inputs = [
    gr.Textbox(label="pos", value="0"),
    gr.Textbox(label="vel", value="0"),
    gr.Textbox(label="angle", value="0"),
    gr.Textbox(label="ang_vel", value="0")
]

outputs = [
    gr.Textbox(label="Action (0 or 1)"),
    gr.Textbox(label="Q-values")
]

gr.Interface(
    fn=predict_action,
    inputs=inputs,
    outputs=outputs,
    title="DQN CartPole Demo",
    description="Enter CartPole state (pos, vel, angle, ang_vel) to get the agent’s action and Q-values."
).launch()

