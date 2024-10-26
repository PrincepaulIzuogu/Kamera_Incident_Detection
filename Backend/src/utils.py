from twilio.rest import Client

def send_alert(message):
    # Twilio credentials (replace with your actual credentials)
    account_sid = 'your_account_sid'
    auth_token = 'your_auth_token'
    client = Client(account_sid, auth_token)

    # Send SMS
    client.messages.create(
        body=message,
        from_='+1234567890',  # Your Twilio number
        to='+0987654321'      # Your target number
    )
    print("Alert sent:", message)

if __name__ == "__main__":
    send_alert("Test alert: A fall has been detected.")
