const hrLine = document.createElement("hr");
hrLine.classList.add(GPTCHATHR)

/**GPTOption함수 + setTimeout 0.5초
 * 예) STOptionFnc(chatModeChooseSLT)
 */
function STOptionFnc(Fnc){
    setTimeout(()=> addGPTOptions(Fnc),500);
}
function STGPTBblFnc(msg){
    setTimeout(()=> addGPTChattingBubble(msg),500);
}
/**CP(CurrentProcess)에 들어갈 수 있는 string
 *  "모드 선택", "질문 답변", "강아지 고양이", "품종 입력",
 *  "이름 입력", "성별 입력", "생일 입력", "과거병력 입력",
 *  "이미지 진단", "사진 촬영", "추가 질문", "어떤게 궁금"
 */
let CP = "asd"
let AnimalName = ""
let AnimalBirth = ""
let AnimalPastDisease = ""
let AnimalSpecies = ""
let AnimalVariety = ""
let recentUserMessage = ""

/** btnname.addEventListener로 메세지를 출력해주는 갓갓함수*/
function BTNBUBBLEEVENT(btnname){
    btnname.addEventListener("click",()=>{
        addUserChattingBubble(btnname.innerText)
    })
}

//모드 선택
function chatModeChooseSLT(textDiv){
const btn1 = document.createElement("button");
const btn2 = document.createElement("button");
const btn3 = document.createElement("button");
btn1.classList.add(GPTOPTIONBTN, "questionMode")
btn2.classList.add(GPTOPTIONBTN, "imageDiagnose")
btn1.innerText = "질문 모드"
btn2.innerText = "진단 모드"
textDiv.innerText = "원하시는 모드를 선택해 주세요!";
messages.push({ role: 'assistant',content:textDiv.innerText})

textDiv.appendChild(hrLine)

emphasizeKeywords(textDiv)

textDiv.appendChild(btn1)
textDiv.appendChild(btn2)

BTNBUBBLEEVENT(btn1)
BTNBUBBLEEVENT(btn2)
displayContainerScrollTop()
chattingPush(addGPTOptions,chatModeChooseSLT)
CP = "모드 선택";
}

//질문 답변
function youShouldQnASLT(textDiv){
const btn1 = document.createElement("button");
const btn2 = document.createElement("button");
btn1.classList.add(GPTOPTIONBTN, "QnAOK")
btn2.classList.add(GPTOPTIONBTN, "RESETSELECT")
btn1.innerText = "네"
btn2.innerText = "처음으로"
textDiv.innerText = "반려동물의 상태를 알아보기 위해 \n 질문에 대한 답변을 해주세요!"
messages.push({ role: 'assistant',content:textDiv.innerText})

textDiv.appendChild(hrLine)

emphasizeKeywords(textDiv)

textDiv.appendChild(btn1)
textDiv.appendChild(btn2)

BTNBUBBLEEVENT(btn1)
BTNBUBBLEEVENT(btn2)
displayContainerScrollTop()
chattingPush(addGPTOptions,chatModeChooseSLT)
CP = "진단 모드"
}

//강아지야 고양이야
function dogOrCatSLT(textDiv){
    const btn1 = document.createElement("button");
    const btn2 = document.createElement("button");
    btn1.classList.add(GPTOPTIONBTN, "dogBtn")
    btn2.classList.add(GPTOPTIONBTN, "catBtn")
    btn1.innerText = "강아지에요"
    btn2.innerText = "고양이에요"
    textDiv.innerText = "아이가 강아지인지 고양이인지 알려주세요."
    messages.push({ role: 'assistant',content:textDiv.innerText})

    textDiv.appendChild(hrLine)
    
    emphasizeKeywords(textDiv)
    
    textDiv.appendChild(btn1)
    textDiv.appendChild(btn2)
    
    BTNBUBBLEEVENT(btn1)
    BTNBUBBLEEVENT(btn2)
    chattingPush(addGPTOptions,dogOrCatSLT)
    displayContainerScrollTop()
    
    CP = "강아지 고양이"
    }

//성별이 뭡니까
function maleOrFemaleSLT(textDiv){
    const btn1 = document.createElement("button");
    const btn2 = document.createElement("button");
    btn1.classList.add(GPTOPTIONBTN, "male")
    btn2.classList.add(GPTOPTIONBTN, "female")
    btn1.innerText = "수컷이에요"
    btn2.innerText = "암컷이에요"
    textDiv.innerText = "아이의 성별을 알려주세요."
    messages.push({ role: 'assistant',content:textDiv.innerText})

    textDiv.appendChild(hrLine)
    
    emphasizeKeywords(textDiv)
    
    textDiv.appendChild(btn1)
    textDiv.appendChild(btn2)
    
    BTNBUBBLEEVENT(btn1)
    BTNBUBBLEEVENT(btn2)
    chattingPush(addGPTOptions,maleOrFemaleSLT)
    displayContainerScrollTop()
    
    CP = "성별 입력"
    }

//아이의 진단이 끝났다
function petDiagnoseCompleteSLT(textDiv){
const btn1 = document.createElement("button");
const btn2 = document.createElement("button");
btn1.classList.add(GPTOPTIONBTN, "moreQuestion")
btn2.classList.add(GPTOPTIONBTN, "RESETSELECT")
btn1.innerText = "추가 질문하기"
btn2.innerText = "처음으로"
textDiv.innerText = "아이의 진단이 끝났습니다!\n추가 질문을 원하시면 선택해주세요."

textDiv.appendChild(hrLine)

emphasizeKeywords(textDiv)

textDiv.appendChild(btn1)
textDiv.appendChild(btn2)

BTNBUBBLEEVENT(btn1)
BTNBUBBLEEVENT(btn2)

chattingPush(addGPTOptions,petDiagnoseCompleteSLT)
displayContainerScrollTop()
CP = "추가 질문"
}

//이미지 진단 모드로 질병을 분석할까요?
function imageDiagnoseSLT(textDiv){
const btn1 = document.createElement("button");
const btn2 = document.createElement("button");
btn1.classList.add(GPTOPTIONBTN, "cameraReadyOn")
btn2.classList.add(GPTOPTIONBTN, "RESETSELECT")
btn1.innerText = "이미지 진단 모드 사용"
btn2.innerText = "처음으로"
textDiv.innerText = "이미지 진단 모드로 질병을 분석할까요?"
messages.push({ role: 'assistant',content:textDiv.innerText})

textDiv.appendChild(hrLine)

emphasizeKeywords(textDiv)

textDiv.appendChild(btn1)
textDiv.appendChild(btn2)

BTNBUBBLEEVENT(btn1)
BTNBUBBLEEVENT(btn2)
chattingPush(addGPTOptions,imageDiagnoseSLT)
displayContainerScrollTop()
CP = "이미지 진단"
}

//사진 재업로드
function imageReuproadSLT(textDiv){
    const btn1 = document.createElement("button");
    const btn2 = document.createElement("button");
    btn1.classList.add(GPTOPTIONBTN, "immediateDiagnose")
    btn2.classList.add(GPTOPTIONBTN, "RESETSELECT")
    btn1.innerText = "진단하기"
    btn2.innerText = "다시 업로드"
    textDiv.innerText = "업로드하신 사진을 통해 진단하시겠어요?"
    messages.push({ role: 'assistant',content:textDiv.innerText})
    
    textDiv.appendChild(hrLine)
    
    emphasizeKeywords(textDiv)
    
    textDiv.appendChild(btn1)
    textDiv.appendChild(btn2)
    
    BTNBUBBLEEVENT(btn1)
    BTNBUBBLEEVENT(btn2)
    
    chattingPush(addGPTOptions,petDiagnoseCompleteSLT)
    displayContainerScrollTop()
    CP = "재업로드"
    }

//사진을 촬영하시겠습니까?
function takePictureSLT(textDiv){
const btn1 = document.createElement("button");
btn1.classList.add(GPTOPTIONBTN, "cameraOn")
btn1.innerText = "사진 업로드"
textDiv.innerText = "진단을 위해 사진을 촬영하시겠습니까?"
messages.push({ role: 'assistant',content:textDiv.innerText})

textDiv.appendChild(hrLine)

emphasizeKeywords(textDiv)  

textDiv.appendChild(btn1)

BTNBUBBLEEVENT(btn1)
chattingPush(addGPTOptions,takePictureSLT)
displayContainerScrollTop()
CP = "사진 업로드"
}

//추가 질문 있으심까
function moreQuestionSLT(textDiv){
const btn1 = document.createElement("button");
const btn2 = document.createElement("button");
const btn3 = document.createElement("button");
btn1.classList.add(GPTOPTIONBTN, "expertFeedbackRequest")
btn2.classList.add(GPTOPTIONBTN, "similarCases")
btn3.classList.add(GPTOPTIONBTN, "recommendHospital")
btn1.innerText = "전문가 상담 요청"
btn2.innerText = "유사 사례"
btn3.innerText = "병원 추천"
textDiv.innerText = "어떤 것이 궁금하시나요?\n이 외의 질문은 채팅에 남겨주세요."
messages.push({ role: 'assistant',content:textDiv.innerText})
textDiv.appendChild(hrLine)

emphasizeKeywords(textDiv)

textDiv.appendChild(btn1)
textDiv.appendChild(btn2)
textDiv.appendChild(btn3)

BTNBUBBLEEVENT(btn1)
BTNBUBBLEEVENT(btn2)
BTNBUBBLEEVENT(btn3)
chattingPush(addGPTOptions,moreQuestionSLT)
displayContainerScrollTop()
CP = "어떤게 궁금"
}



/** 파라미터 안에 원하는 함수 넣으면 모드 선택해주는 콜백함수,
 * chatModeChoose, youShouldQnA, imageDiagnoseSLT, takePictureSLT, petDiagnoseCompleteSLT, moreQuestionSLT
 */
function addGPTOptions(selectFncs){
const boxDiv = document.createElement("div");
const textDiv = document.createElement("div");

boxDiv.classList.add(GPTCHATBOX)
textDiv.classList.add(GPTCHATTEXT)

displayContainer.appendChild(boxDiv)

gptHeadChecker(boxDiv);

boxDiv.appendChild(textDiv)
selectFncs(textDiv)

recentSpeaker = "gpt";
}

//user말풍선 안에 사진넣기
function addUserImageBubble(src){
const boxDiv = document.createElement("div");
const img = document.createElement("img");

boxDiv.classList.add(USERCHATBOX)
img.src = src;
img.classList.add(USERCHATTEXT)
img.style.padding = "0px";
displayContainer.appendChild(boxDiv)

boxDiv.appendChild(img)



recentSpeaker = "user";
chattingPush(addUserImageBubble,src)

displayContainerScrollTop()
}