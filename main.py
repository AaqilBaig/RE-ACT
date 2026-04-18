import requests
import json
import time
import os
import speech_recognition as sr
from dotenv import load_dotenv
from obj_det import locate_and_segment

load_dotenv()

def get_voice_command():
    """
    Uses the system microphone to capture a voice command and translates it to text.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n[Microphone] Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("\n[Microphone] Recording for 5 seconds... Speak now!")
        try:
            # Force record for exactly 5 seconds (bypasses silence-detection bugs where it hangs)
            audio = recognizer.record(source, duration=5)
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
    print("[LLM Processing] Thinking...")
    
    response = requests.post(
      url="https://openrouter.ai/api/v1/chat/completions",
      headers={
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
      },
      data=json.dumps({
        "model": "arcee-ai/trinity-large-preview:free",
        "messages": [
          {
            "role": "user",
            "content": prompt_text
          }
        ]
      })
    )

    try:
        data = response.json()
        message = data['choices'][0]['message']['content'].strip()
        
        # Clean up the message if returned with markdown code blocks
        if "```json" in message:
            message = message.split("```json")[1].split("```")[0].strip()
        elif "```" in message:
            message = message.split("```")[1].strip()
            
        plan = json.loads(message)
        print("\n[LLM Response] Plan generated successfully:")
        print(json.dumps(plan, indent=2))
        return plan
        
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM JSON: {e}")
        print(f"Raw Output: {message}")
        return []
    except Exception as e:
        print(f"Error connecting to LLM: {e}")
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
                print(f"[Robot Hardware] Moving arm to coordinates: X:{coords[0]}, Y:{coords[1]}")
                time.sleep(1)
                print(f"[Robot Hardware] Closing gripper. '{target}' grasped successfully.")
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
