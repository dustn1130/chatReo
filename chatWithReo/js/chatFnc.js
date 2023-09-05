const chattingName = document.querySelector("#chattingName")
const displayContainer = document.querySelector(".displayContainer")
const chatInput = document.querySelector(".chatReoChattingTextSpace")
const sendButton = document.querySelector(".chatReoChattingBtn")

const USERCHATBOX = "userChattingBox"
const USERCHATTEXT = "userChattingText"
const GPTCHATBOX = "gptChattingBox"
const GPTCHATTEXT = "gptChattingText"
const GPTCHATHR = "gptChattingHr"
const GPTOPTIONBTN = "gptOptionBtns"

const GPTIMGSRC = "chatReoImage/chatReoGPTProfile.svg"

let recentSpeaker = ""

let disablePoint = 0;
let emphasizeDisable = 0;

let chatObject = {
    bubbleForm:"",
    value:""
};

let chatList = [];

const emphasizeKey = 
['당신의 반려동물의 건강 상태를 진단',
 '해당 질병에 대해 도움',
'원하시는 모드를 선택',
'질문에 대한 답변',
'이미지 진단 모드',
'사진을 촬영',
'업로드',
'강아지',
'고양이',
'품종',
'이름',
'성별',
'생년월일',
'과거 병력',
'사진 촬영',
'아이']

//displayContainer스크롤 맨아래로 내리기
function displayContainerScrollTop(){
    if (disablePoint == 1){
        return(false)
    }
    displayContainer.scrollTop = displayContainer.scrollHeight;
}
//채팅list push하기
function chattingPush(bubForm,text){
    if (disablePoint == 1){
        return(false)
    }
    const chatObject = {

        bubbleForm:bubForm,
        value:text
    }
    chatList.push(chatObject)
}


//user 채팅치면 말풍선 만들기
function addUserChattingBubble(text){
    if (text == ""){
        return
    }
    const boxDiv = document.createElement("div");
    const textDiv = document.createElement("div");
    messages.push({ role: 'user',content:text})

    boxDiv.classList.add(USERCHATBOX)
    textDiv.innerText = text;
    textDiv.classList.add(USERCHATTEXT)

    displayContainer.appendChild(boxDiv)
    boxDiv.appendChild(textDiv)
    chatInput.value = "";
    
    recentSpeaker = "user";
    recentUserMessage = textDiv.innerText;
    
    displayContainerScrollTop()
    chattingPush(addUserChattingBubble,text)
    GPTLogicFnc()
}

//gpt말풍선 강조
function emphasizeKeywords(div){
    const text = div.innerHTML;
    const newText = emphasizeKey.reduce((text, keyword)=>{
        const emphasizedText = `<span class="emphasizedText">${keyword}</span>`;
        return text.replace(new RegExp(keyword,'g'), emphasizedText);

    },text)
    div.innerHTML = newText;
}

//gpt대가리 만들기
function gptHeadChecker(boxDiv){
    if (recentSpeaker != "gpt"){
        const gptProfileDiv = document.createElement("div");
        const gptImgElement = document.createElement("img");
        
        gptProfileDiv.classList.add("gptProfilePosition")
        gptImgElement.src = GPTIMGSRC;
        gptImgElement.alt = "gpt얼굴"
        
        boxDiv.appendChild(gptProfileDiv)
        gptProfileDiv.appendChild(gptImgElement);
    }
}

//gpt말풍선 만들기
function addGPTChattingBubble(text){

    messages.push({role:'assistant', content: text})
    const boxDiv = document.createElement("div");
    const textDiv = document.createElement("div");

    boxDiv.classList.add(GPTCHATBOX)
    textDiv.innerText = text;
    textDiv.classList.add(GPTCHATTEXT)

    displayContainer.appendChild(boxDiv)

    gptHeadChecker(boxDiv);

    boxDiv.appendChild(textDiv)
    if (emphasizeDisable == 0){
        emphasizeKeywords(textDiv)
    }
    recentSpeaker = "gpt";
    recentMessage = textDiv.innerText;
    chattingPush(addGPTChattingBubble,text)
    displayContainerScrollTop()
}

//chatList에 있는 오브젝트들대로 말풍선을 출력해드립니다
function chatListPrinter(chatList){
    disablePoint = 1;
    for (let i = 0; i < chatList.length;i++){
        (chatList[i].bubbleForm)(chatList[i].value)
    }
    disablePoint = 0;
    displayContainerScrollTop()
}