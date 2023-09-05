"use strict"

//입력 시 닉네임, 메세지로 채팅 보내기
sendButton.addEventListener("click",()=>{
    event.preventDefault();
    addUserChattingBubble(chatInput.value)
})
