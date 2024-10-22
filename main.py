from dotenv import load_dotenv
import os


if __name__=="__main__":
    print("running main")
    # get environment variable
    print("getting environment variable")
    load_dotenv()
    print("Test=")
    print(os.getenv("TEST"))
    print("buildtag=1")
    print(os.getenv("buildtag"))
    
    
    
else:
    print("not running main")
    
    