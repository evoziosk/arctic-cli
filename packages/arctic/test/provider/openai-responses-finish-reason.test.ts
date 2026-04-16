import { describe, expect, test } from "bun:test"
import { mapOpenAIResponseFinishReason } from "../../src/provider/sdk/openai-compatible/src/responses/map-openai-responses-finish-reason"

describe("mapOpenAIResponseFinishReason", () => {
  test("returns tool-calls for normal completion when client tool calls happened", () => {
    expect(
      mapOpenAIResponseFinishReason({
        finishReason: undefined,
        hasFunctionCall: true,
      }),
    ).toBe("tool-calls")
  })

  test("returns stop for normal completion without client tool calls", () => {
    expect(
      mapOpenAIResponseFinishReason({
        finishReason: undefined,
        hasFunctionCall: false,
      }),
    ).toBe("stop")
  })

  test("returns explicit incomplete reasons when provided", () => {
    expect(
      mapOpenAIResponseFinishReason({
        finishReason: "max_output_tokens",
        hasFunctionCall: true,
      }),
    ).toBe("length")

    expect(
      mapOpenAIResponseFinishReason({
        finishReason: "content_filter",
        hasFunctionCall: true,
      }),
    ).toBe("content-filter")

    expect(
      mapOpenAIResponseFinishReason({
        finishReason: "tool_calls",
        hasFunctionCall: true,
      }),
    ).toBe("tool-calls")
  })

  test("returns unknown for unknown reason without client tool calls", () => {
    expect(
      mapOpenAIResponseFinishReason({
        finishReason: "completed",
        hasFunctionCall: false,
      }),
    ).toBe("unknown")
  })
})
