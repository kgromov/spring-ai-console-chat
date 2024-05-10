package org.kgromov;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.ChatClient;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingClient;
import org.springframework.ai.reader.ExtractedTextFormatter;
import org.springframework.ai.reader.pdf.PagePdfDocumentReader;
import org.springframework.ai.reader.pdf.config.PdfDocumentReaderConfig;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.ApplicationRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.core.io.Resource;
import org.springframework.util.StopWatch;

import java.util.Map;
import java.util.Scanner;

import static java.util.stream.Collectors.joining;

@Slf4j
@SpringBootApplication
public class SpringAiConsoleChatApplication {

    public static void main(String[] args) {
        SpringApplication.run(SpringAiConsoleChatApplication.class, args);
    }

    @Bean
    VectorStore vectorStore(EmbeddingClient embeddingClient,
                            @Value("classpath:${settings.input.source}") Resource pdfResource) {
        var vectorStore = new SimpleVectorStore(embeddingClient);
        var config = PdfDocumentReaderConfig.builder()
                .withPageExtractedTextFormatter(new ExtractedTextFormatter.Builder()
                        .withNumberOfBottomTextLinesToDelete(0)
                        .withNumberOfTopPagesToSkipBeforeDelete(0)
                        .build())
                .withPagesPerDocument(1)
                .build();
        var pdfReader = new PagePdfDocumentReader(pdfResource, config);
        var textSplitter = new TokenTextSplitter();
        vectorStore.accept(textSplitter.apply(pdfReader.get()));
        return vectorStore;
    }

    @Bean
    ApplicationRunner interactiveChatRunner(VectorStore vectorStore,
                                            ChatClient chatClient,
                                            @Value("classpath:/prompts/prompt-template.st") Resource promptTemplate) {
        return args -> {
            log.info("Spin up chat");
            try(Scanner scanner = new Scanner(System.in)) {
                while (true) {
                    System.out.print("User: ");
                    String question = scanner.nextLine();

                    if ("exit".equalsIgnoreCase(question)) {
                        break;
                    }
                    StopWatch stopWatch = new StopWatch();
                    stopWatch.start("Answer question");
                    try {
                        var documents = vectorStore.similaritySearch(SearchRequest.query(question).withTopK(3));
                        String documentsContent = documents.stream().map(Document::getContent).collect(joining("\n"));
                        var template = new PromptTemplate(promptTemplate);
                        Prompt prompt = template.create(
                                Map.of(
                                        "input", question,
                                        "documents", documentsContent
                                )
                        );
                        String answer = chatClient.call(prompt)
                                .getResult()
                                .getOutput()
                                .getContent();

                        System.out.println("Assistant: " + answer);
                    } finally {
                        stopWatch.stop();
                        var taskInfo = stopWatch.lastTaskInfo();
                        log.info("Time to {} = {} ms", taskInfo.getTaskName(), taskInfo.getTimeMillis());
                    }
                }
            }
        };
    }
}
