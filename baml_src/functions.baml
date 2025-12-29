// Judge Conversation Quality
// Evaluates AI coding assistant conversations for training data quality

function JudgeConversation(conversation: string) -> ConversationScore {
  client OpenRouterClient

  prompt #"
    Evaluate this AI coding assistant conversation. Rate 1-5 for each criterion.

    Conversation:
    ---
    {{ conversation }}
    ---

    Criteria:
    - task_completion: Did assistant complete the request? (1=failed, 5=perfect)
    - tool_use: Were tools used effectively? (1=poor, 5=excellent)
    - response_quality: Were responses clear and helpful? (1=poor, 5=excellent)
    - code_quality: Was code correct? (1=poor, 5=excellent, 3=no code)

    Provide scores and a brief reasoning explaining your evaluation.

    {{ ctx.output_format }}
  "#
}
