import requests
import os
from dotenv import load_dotenv
load_dotenv()


def notify(message, event="coegan_epoch_ended", ifttt_key=os.getenv("IFTTT_WEBHOOK_KEY")):
    if ifttt_key is None:
        return
    report = {'value1': message}
    requests.post(f"https://maker.ifttt.com/trigger/{event}/with/key/{ifttt_key}", data=report)
