const conversationData = [
    {
        type: "chatting",
        text: "안녕하세요 닥터 레오에요. 반가워요!"
    },
    {
        type: "chatting",
        text: "저는 당신의 반려동물의 건강 상태를 진단하고 해당 질병에 대해 도움을 줄 수 있어요.\n필요한 정보가 있다면 제가 제공해드릴게요!"
    },
    {
        type: "options",
        function: chatModeChooseSLT
    }
    ];
    
    function simulateChat(dataArray, index) {
    if (index < dataArray.length) {
        const data = dataArray[index];
    
        if (data.type === "chatting") {
            addGPTChattingBubble(data.text);
        } else if (data.type === "options") {
            addGPTOptions(data.function);
        }
    
        setTimeout(() => {
            simulateChat(dataArray, index + 1);
        }, 500);
    }
    }
    
    /**CP(현재단계)랑 recentmsg(최근메세지) 입력하면 체크해드립니다 */
    function CPRctMsgChker(cpmsg, stringArray) {
        if (CP != cpmsg){
            return (false);
        }
        if ((typeof stringArray)=='string'){
            if(recentUserMessage.includes(stringArray)){
                return (true);
            }
        }
        if ((typeof stringArray)=='object'){for (let i = 0; i < stringArray.length; i++) {
            
            if (recentUserMessage.includes(stringArray[i])) {
                return (true);
            }
        }}
        return (false)
    
    }
    /**유저가 채팅창 입력했을때 다음 입력 제시해주는 로직함수, */
    function GPTLogicFnc(){
        if (disablePoint == 1){
            return
        }
        console.log(CP)
        if (CPRctMsgChker("모드 선택",["진단모드","진단 모드"])){
            STOptionFnc(youShouldQnASLT)
            CP = "진단 모드"
        }else if (recentUserMessage == "홈으로" || recentUserMessage == "처음으로"){
            STOptionFnc(chatModeChooseSLT)
            CP = "모드 선택"
        } 
        else if (CPRctMsgChker("모드 선택",["질문 모드","질문모드"])){
            CP = "질문 모드"
            STGPTBblFnc("어떤게 궁금하신가요?")
        } else if (CP == "질문 모드"){
            chatGPT(recentUserMessage)
        }
         else if (CPRctMsgChker("진단 모드","네")){
            STOptionFnc(dogOrCatSLT)
            CP = "강아지 고양이"
        } else if (CPRctMsgChker("강아지 고양이","강아지")){
            STGPTBblFnc("강아지의 품종을 알려주세요.")
            CP = "품종 입력"
        } else if (CPRctMsgChker("강아지 고양이","고양이")){
            STGPTBblFnc("고양이의 품종을 알려주세요.")
            CP = "품종 입력"
        } else if (CP == "품종 입력"){
            STOptionFnc(maleOrFemaleSLT)
            CP = "성별 입력"
        } else if (CPRctMsgChker("성별 입력",["남자","수컷","남성"])){
            STGPTBblFnc("아이의 생일을 알려주세요.\n예)21년 8월 21일")
            CP = "생일 입력"
        } else if (CPRctMsgChker("성별 입력",["여자","암컷","여성"])){
            STGPTBblFnc("아이의 생일을 알려주세요.\n예)21년 8월 21일")
            CP = "생일 입력"
        } else if (CP == "생일 입력"){
            STGPTBblFnc("아이의 과거 병력이 있다면, 알려주세요.")
            CP = "과거병력 입력"
        } else if (CP == "과거병력 입력"){
            STGPTBblFnc("아이의 어떤 부위에 대해 진단받고 싶은 지 알려주세요.")
            CP = "지금부터 진단 GPT에게 맡긴다 모드"
        } else if (CP == "지금부터 진단 GPT에게 맡긴다 모드"){
            diagGPT()
        }
    
        else if (CPRctMsgChker("추가 질문",["추가","질문","물어","여쭤"])){
            CP = "질문 모드"
            STGPTBblFnc("어떤게 궁금하신가요?")
        }
    
    }
    
    simulateChat(conversationData, 0);