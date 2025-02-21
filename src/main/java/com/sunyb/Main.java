package com.sunyb;

import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.ModelSupport;
import com.github.tjake.jlama.model.functions.Generator;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.prompt.PromptContext;
import com.github.tjake.jlama.util.Downloader;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.jlama.JlamaChatModel;
import lombok.SneakyThrows;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.UUID;

/**
 * @author yb.Sun
 * @date 2025/2/21 13:51
 */
public class Main {
    public static void main(String[] args) {
        chat();
        sample();
    }

    @SneakyThrows
    public static void sample() {
        String model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B";
        String workingDirectory = "./models";

        String prompt = "What is the best season to plant avocados?";

        // Downloads the model or just returns the local path if it's already downloaded
        File localModelPath = new Downloader(workingDirectory, model).huggingFaceModel();

        // Loads the quantized model and specified use of quantized memory
        AbstractModel m = ModelSupport.loadModel(localModelPath, DType.F32, DType.I8);

        PromptContext ctx;
        // Checks if the model supports chat prompting and adds prompt in the expected format for this model
        if (m.promptSupport().isPresent()) {
            ctx = m.promptSupport()
                    .get()
                    .builder()
                    .addSystemMessage("你的名字叫贾维斯, 你是一个幽默、风趣的人. 你喜欢唱、跳、rap, 你的练习时长是2.5年; 你的所有回答都是用中文进行回复.")
                    .addUserMessage("你叫什么名字?")
                    .build();
        } else {
            ctx = PromptContext.of(prompt);
        }

        System.out.println("Prompt: " + ctx.getPrompt() + "\n\n");
        // Generates a response to the prompt and prints it
        // The api allows for streaming or non-streaming responses
        // The response is generated with a temperature of 0.7 and a max token length of 256
        StringBuilder sb = new StringBuilder();
        System.out.println("\n");
        System.out.println("\n----------思考开始----------\n");
        Generator.Response r = m.generate(UUID.randomUUID(), ctx, 0.7f, 8192, (s, f) -> {
            sb.append(s);
            System.out.print(s);
        });
        System.out.println("\n----------思考结束----------\n");
        System.out.println("\n");
        System.out.println("思考结果: " + sb.toString());
        System.out.println("\n");
        System.out.println("\n----------参数r----------\n");
        System.out.println("r: = " + r.toString());
        System.out.println("\n");
        System.out.println("\n----------结果----------\n");
        System.out.println(r.responseText);
    }

    public static void chat() {
        ChatLanguageModel model = JlamaChatModel.builder()
                .modelCachePath(Path.of("./models"))
                .modelName("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
                .maxTokens(8192)
                .temperature(0.7f)
                .build();

        ChatResponse chatResponse = model.chat(
                SystemMessage.from("你的名字叫贾维斯, 你是一个幽默、风趣的人. 你喜欢唱、跳、rap, 你的练习时长是2.5年; 你的所有回答都是用中文进行回复."),
                UserMessage.from("你叫什么名字?")
        );

        System.out.println("\n" + chatResponse.aiMessage().text() + "\n");
    }
}