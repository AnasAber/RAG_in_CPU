import React, { useState, useRef } from 'react';
import { Upload, MessageSquare, Send, Trash } from 'lucide-react'; // Import Trash icon
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Card, CardHeader, CardTitle, CardContent } from './components/ui/card';

const RAGInterface = () => {
  const [file, setFile] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileUpload = async (event) => {
    const uploadedFile = event.target.files[0];
    setFile(uploadedFile);

    const formData = new FormData();
    formData.append('file', uploadedFile);

    try {
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      console.log(data.message);
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  const handleSendMessage = async () => {
    if (inputMessage.trim() === '') return;

    const newMessage = { text: inputMessage, sender: 'user' };
    setMessages([...messages, newMessage]);
    setInputMessage('');

    try {
      const response = await fetch('http://localhost:5000/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: inputMessage }),
      });
      const data = await response.json();
      simulateTypingEffect(data.response);
    } catch (error) {
      console.error('Error sending message:', error);
    }
  };

  const simulateTypingEffect = (botResponse) => {
    setIsTyping(true);
    let currentIndex = 0;
    const typingInterval = setInterval(() => {
      if (currentIndex < botResponse.length) {
        const botText = botResponse.slice(0, currentIndex + 15);
        const typingMessage = { text: botText, sender: 'bot' };
        setMessages((prevMessages) => {
          // Replace the last bot message with the "typing" one
          const newMessages = [...prevMessages];
          if (newMessages.length > 0 && newMessages[newMessages.length - 1].sender === 'bot') {
            newMessages[newMessages.length - 1] = typingMessage;
          } else {
            newMessages.push(typingMessage);
          }
          return newMessages;
        });
        currentIndex++;
      } else {
        clearInterval(typingInterval);
        setIsTyping(false);
      }
    }, 10); // Adjust typing speed by changing the interval time
  };

  const handleNewCV = async () => {
    // Clear messages
    setMessages([]);
    setFile(null);

    // Make an API call to clear ChromaDB data
    try {
      const response = await fetch('http://localhost:5000/clear_cv_data', {
        method: 'POST',
      });
      const data = await response.json();
      console.log(data.message);
    } catch (error) {
      console.error('Error clearing CV data:', error);
    }
  };

  return (
    <div className="flex flex-col h-screen max-w-2xl mx-auto p-4">
      <Card className="mb-4">
        <CardHeader>
          <CardTitle>File Upload</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center">
            <Input
              type="file"
              ref={fileInputRef}
              className="hidden"
              onChange={handleFileUpload}
            />
            <Button onClick={() => fileInputRef.current.click()}>
              <Upload className="mr-2 h-4 w-4" /> Upload File
            </Button>
            {file && <span className="ml-2">{file.name}</span>}
          </div>
        </CardContent>
      </Card>

      <Card className="flex-grow flex flex-col">
        <CardHeader>
          <CardTitle>Chat</CardTitle>
        </CardHeader>
        <CardContent className="flex-grow overflow-y-auto">
          {messages.map((message, index) => (
            <div key={index} className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'} mb-2`}>
              <div className={`p-2 rounded-lg ${message.sender === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}>
                {message.text}
              </div>
            </div>
          ))}
          {/* {isTyping && (
            <div className="flex justify-start mb-2">
              <div className="p-2 rounded-lg bg-gray-200">Typing...</div>
            </div>
          )} */}
        </CardContent>
        <div className="p-4 border-t">
          <div className="flex items-center">
            <Input
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Type your message..."
              className="flex-grow mr-2"
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
            />
            <Button onClick={handleSendMessage}>
              <Send className="h-4 w-4" />
            </Button>
            <Button variant="destructive" onClick={handleNewCV}>
              <Trash className="h-4 w-4" /> 
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default RAGInterface;
