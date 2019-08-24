import agent

def main():
    
    print("\nWelcome to MountainCar RL:")

    while True:
        user = input("1. Train\n2. Env demo\n3. Run agent\n4. Quit\n")
        
        if (user == "1"):
            train_model()
        elif (user == "2"):
            pass
        elif (user == "3"):
            pass
        elif (user == "4"):
            break

def train_model():
    a = agent.agent()
    a.train()

main()
    