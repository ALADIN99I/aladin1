class CommunicationBus:
    def __init__(self):
        self.messages = []

    def post_message(self, sender, receiver, message):
        """Posts a message to the bus."""
        self.messages.append({
            "sender": sender,
            "receiver": receiver,
            "message": message
        })

    def get_messages(self, receiver):
        """Gets all messages for a specific receiver."""
        return [msg for msg in self.messages if msg['receiver'] == receiver]
