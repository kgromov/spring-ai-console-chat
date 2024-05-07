package org.kgromov;

import org.springframework.ai.chat.ChatClient;
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
import org.springframework.context.annotation.DependsOn;
import org.springframework.core.io.Resource;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.stream.Collectors;

@SpringBootApplication
public class SpringAiConsoleChatApplication {
    @Value("classpath:${settings.input.source}")
    private Resource resource;

    public static void main(String[] args) {
        SpringApplication.run(SpringAiConsoleChatApplication.class, args);
    }

    @Bean
    VectorStore vectorStore(EmbeddingClient embeddingClient) {
        return new SimpleVectorStore(embeddingClient);
    }

    void init(VectorStore vectorStore, Resource pdfResource) {
        var config = PdfDocumentReaderConfig.builder()
                .withPageExtractedTextFormatter(new ExtractedTextFormatter.Builder().withNumberOfBottomTextLinesToDelete(3)
                        .withNumberOfTopPagesToSkipBeforeDelete(1)
                        .build())
                .withPagesPerDocument(1)
                .build();
        var pdfReader = new PagePdfDocumentReader(pdfResource, config);
        var textSplitter = new TokenTextSplitter();
        vectorStore.accept(textSplitter.apply(pdfReader.get()));
    }

    @Bean
    ApplicationRunner interactiveChatRunner(VectorStore vectorStore, ChatClient chatClient) {
        return args -> {
            System.out.println("Spin up chat");
            Scanner scanner = new Scanner(System.in);
            while (true) {
                System.out.print("User: ");
                String question = scanner.nextLine();

                if ("exit".equalsIgnoreCase(question)) {
                    break;
                }
                this.init(vectorStore, resource);
                PromptTemplate promptTemplate = new PromptTemplate(question);
                String answer = chatClient.call(promptTemplate.create())
                        .getResult()
                        .getOutput()
                        .getContent();
                System.out.println("Agent: " + answer);
            }
            scanner.close();
        };
    }

}
