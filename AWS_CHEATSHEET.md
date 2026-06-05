# AWS Resumption Cheat Sheet

When you return from the demo, you don't need to guess how to get back in. We have everything documented right here!

## 1. Wake Up the Server
1. Go to your [AWS EC2 Dashboard](https://eu-north-1.console.aws.amazon.com/ec2/home?region=eu-north-1#Instances:).
2. Make sure your region (top right) is set to **Europe (Stockholm) eu-north-1**.
3. Check the box next to your `BCREC` server.
4. Click **Instance state** at the top right -> **Start instance**.

## 2. Get Your New IP Address
*Note: Every time you stop an AWS server on the Free Tier, it gives you a brand new Public IP address when you start it back up! The old one (`56.228.35.149`) is gone forever.*

1. Wait for the Instance State to say **Running**.
2. Look at the Details pane at the bottom of the screen and copy your **new Public IPv4 address**.

## 3. Connect via Terminal
Open **PowerShell** on your laptop and run these two commands:

```powershell
cd C:\Users\ANAMIKA\Downloads
ssh -i "voice-agent-key.pem" ubuntu@PASTE_NEW_IP_HERE
```
*(Type `yes` when it asks about the fingerprint).*

You are now back in!
