// Conversation Quality Score
// Structured output from LLM judge evaluation

class ConversationScore {
  task_completion int @description("Did assistant complete the request? (1=failed, 5=perfect)")
  tool_use int @description("Were tools used effectively? (1=poor, 5=excellent)")
  response_quality int @description("Were responses clear and helpful? (1=poor, 5=excellent)")
  code_quality int @description("Was code correct? (1=poor, 5=excellent, 3=no code)")
  reasoning string @description("Brief explanation of the scores")
}
