import requests
import json
import time
import os
import subprocess
import re
import speech_recognition as sr
import serial
import json
from dotenv import load_dotenv
from obj_det import locate_and_segment

load_dotenv()

# --- ARDUINO SERIAL SETUP ---
ARDUINO_PORT = os.getenv("ARDUINO_PORT", "COM17")
try:
    print(f"[Initialization] Connecting to Arduino on {ARDUINO_PORT}...")
    arduino = serial.Serial(ARDUINO_PORT, 115200, timeout=1)
    time.sleep(2) # Wait for Arduino to reset upon connection
    print("[Initialization] Arduino connected!")
except Exception as e:
    print(f"[Initialization] Failed to connect to Arduino: {e}")
    arduino = None

def pixels_to_mm(px_x, px_y):
    cal_file = "calibration.json"
    if os.path.exists(cal_file):
        with open(cal_file, "r") as f:
            cal = json.load(f)
        
        # Real coordinate = (Pixel - Base Pixel) * Scale
        # Updated axis logic: Camera X -> Physical X, Camera Y -> Physical Y
        # (scale_x will naturally be negative because the X-axis is inverted)
        real_x = (px_x - cal["base_px_x"]) * cal["scale_x"]
        real_y = (px_y - cal["base_px_y"]) * cal["scale_y"]
        
        # Optional constraint for working space:
        if real_y < 86.2: 
            # Cap the minimum Y physically reachable
            print("[Warning] Object is closer to base than the 86.2mm workspace offset! Capping Y.")
            real_y = 86.2
            
        return real_x, real_y
    else:
        print("[Warning] No calibration.json found! Using fallback offsets. Please run calibrate.py first!")
        # Fallback values from previous setup attempt
        real_x = (115 - px_y) * 0.4878
        real_y = 100.0 + (273 - px_x) * 0.4878
        return real_x, real_y

def send_coordinates_to_arduino(x, y, z):
    if not arduino:
        print("[Serial] Arduino not connected. Skipping hardware command.")
        return
        
    print(f"[Serial] Transmitting -> X:{x:.1f}, Y:{y:.1f}, Z:{z:.1f}")
    
    arduino.write(f"{x}\n".encode('utf-8'))
    time.sleep(0.5)
    
    arduino.write(f"{y}\n".encode('utf-8'))
    time.sleep(0.5)
    
    arduino.write(f"{z}\n".encode('utf-8'))
    time.sleep(0.5)
# ----------------------------

def get_voice_command():
    """
    Uses the system microphone to capture a voice command and translates it to text.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n[Microphone] Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("\n[Microphone] Listening... Speak now!")
        try:
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=6)
            print("[Microphone] Processing speech...")
            # Use Google's free Web Speech API (no API key required)
            text = recognizer.recognize_google(audio)
            print(f"[Voice Input] You said: '{text}'")
            return text
        except sr.WaitTimeoutError:
            print("[Microphone] Error: Listening timed out while waiting for phrase to start.")
        except sr.UnknownValueError:
            print("[Microphone] Error: Could not understand audio. Please speak clearly.")
        except sr.RequestError as e:
            print(f"[Microphone] Error: Could not request results; {e}")
    return None

def get_llm_plan(task_description):
    """
    Queries an LLM to break down a high-level task into a structured JSON plan.
    """
    prompt_text = f"""
    Convert the following user task into a structured JSON array of robotic actions.
    Allowed actions: 
    - "grasp" (requires "target" string, which is the physical object name. INCLUDE adjectives like colors or materials if mentioned, e.g., "green tape" or "metallic cup")
    - "move" (requires "target" string, which can be a direction or destination location)
    - "drop" (requires "location" string)
    
    User task: "{task_description}"
    
    Return ONLY valid JSON, without any conversational text or markdown blocks.
    Example output format: 
    [
      {{"action": "grasp", "target": "red apple"}}, 
      {{"action": "move", "target": "basket"}}, 
      {{"action": "drop", "location": "basket"}}
    ]
    """
    
    print(f"\n[LLM Request] Task: {task_description}")
    print("[LLM Processing] Thinking using local llama-cli... (This may take 30-60 seconds on CPU to load the model and generate tokens)")
    
    llama_dir = r"D:\llama\llama-b8884-bin-win-cpu-x64"
    llama_exe = os.path.join(llama_dir, "llama-cli.exe")
    model_path = os.path.join(llama_dir, "gemma-4-E2B-it-Q4_K_M.gguf")

    try:
        process = subprocess.Popen(
            [
                llama_exe,
                "-m", model_path,
                "-n", "2048", # Increased drastically so the thinking process + JSON fits
                "--temp", "0.1",
                "--log-disable", # Suppress verbose llama-cli logs
                "--single-turn", # Run for a single turn then exit, preventing interactive mode hang
                "-p", prompt_text
            ],
            stdin=subprocess.DEVNULL, # Force EOF if it tries to wait for input at "> " prompt
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1
        )
        
        output = ""
        print("\n[LLM Live Generation Start]")
        while True:
            try:
                char = process.stdout.read(1)
                if not char:
                    break
                print(char, end="", flush=True)
                output += char
            except KeyboardInterrupt:
                # Some environments or llama-cli internal states might throw an interrupt when it exits. 
                # We catch it so the program can safely proceed to JSON parsing.
                break
            
        process.wait()
        print("\n[LLM Live Generation End]")
        
        # Clean up the output to extract JSON
        message = output.split(prompt_text)[-1].strip() if prompt_text in output else output.strip()
        
        if "```json" in message:
            message = message.split("```json")[1].split("```")[0].strip()
        elif "```" in message:
            message = message.split("```")[1].strip()
            
        # Try to find JSON array brackets using Regex to ignore [Start thinking] blocks
        match = re.search(r'\[\s*\{.*?\}\s*\]', message, re.DOTALL)
        if match:
            message = match.group(0)
        else:
            # Fallback
            if "[" in message and "]" in message:
                start = message.find("[")
                end = message.rfind("]") + 1
                message = message[start:end]

        plan = json.loads(message)
        print("\n[LLM Response] Plan generated successfully:")
        print(json.dumps(plan, indent=2))
        return plan
        
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM JSON: {e}")
        print(f"Raw Output: {message}")
        return []
    except Exception as e:
        print(f"Error running local LLM: {e}")
        return []

def execute_plan(plan):
    """
    Loops over the structured actions and executes them.
    """
    for step_index, step in enumerate(plan):
        action = step.get("action")
        target = step.get("target") or step.get("location")
        
        print(f"\n--- {step_index+1}. Executing Step: {action.upper()} '{target}' ---")
        
        if action == "grasp":
            # Call vision module to locate the object dynamically at this exact moment
            print(f"Robot locating '{target}' using camera...")
            coords = locate_and_segment(target)
            
            if coords:
                print(f"[Vision] Found '{target}' at pixels -> X:{coords[0]}, Y:{coords[1]}")
                
                # Convert pixels to physical mm offsets
                real_x, real_y = pixels_to_mm(coords[0], coords[1])
                target_z = 20.0 # Define the lowering height for grasping in mm
                
                print(f"[Robot Hardware] Moving arm to physical coordinates: X:{real_x:.1f}, Y:{real_y:.1f}, Z:{target_z:.1f}")
                
                # Dispatch values to Arduino consecutively
                send_coordinates_to_arduino(real_x, real_y, target_z)
                
                print(f"[Robot Hardware] Waiting for grasping sequence to complete...")
                time.sleep(8) # Pause Python orchestration while Arduino physically moves 
            else:
                print(f"[Robot Hardware Error] Could not find '{target}'. Aborting sequence.")
                break
                
        elif action == "move":
            print(f"[Robot Hardware] Moving arm towards '{target}'...")
            time.sleep(1)
            print(f"[Robot Hardware] Position reached.")
            
        elif action == "drop":
            print(f"[Robot Hardware] Opening gripper. Dropping payload into '{target}'.")
            time.sleep(0.5)
            
        else:
            print(f"[Error] Unknown action type: {action}")
            
        time.sleep(1) # Pause between steps

def main():
    print("=== Robot Task Orchestrator Initialized ===")
    
    # 1. Prompt user whether they want to use microphone or typing
    use_voice_input = input("Would you like to use voice commands? (y/n): ").strip().lower()
    
    if use_voice_input == 'y':
        task_description = get_voice_command()
        if not task_description:
            return  # Exit if no command was picked up
    else:
        # Text input fallback
        task_description = input("Enter the task for the robot (e.g., 'pick the green tape only'): ")
        if not task_description.strip():
            print("No task entered. Exiting.")
            return
    
    # 2. Chain-of-Thought / LLM Planning
    plan = get_llm_plan(task_description)
    
    # 3. Execution Loop
    if plan:
        execute_plan(plan)
    
    print("\n=== Task Complete ===")

if __name__ == "__main__":
    main()
