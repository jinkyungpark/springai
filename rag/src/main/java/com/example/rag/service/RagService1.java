package com.example.rag.service;

import java.io.IOException;
import java.util.List;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.SimpleLoggerAdvisor;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.document.Document;
import org.springframework.ai.document.DocumentReader;
import org.springframework.ai.document.DocumentTransformer;
import org.springframework.ai.reader.pdf.PagePdfDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.Ordered;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.Resource;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;
import org.springframework.web.multipart.MultipartFile;

import lombok.extern.slf4j.Slf4j;

@Service
@Slf4j
public class RagService1 {
  // ##### 필드 #####
  private ChatClient chatClient;
  @Autowired private VectorStore vectorStore;
  @Autowired private JdbcTemplate jdbcTemplate;

  // ##### 생성자 #####
  public RagService1(ChatClient.Builder chatClientBuilder) {
    this.chatClient = chatClientBuilder
        .defaultAdvisors(
            // 로그 어드바이저는 가장 낮은 우선순위로 설정해서 다른 어드바이저들이
            // 처리한 후의 최종 프롬프트와 응답을 모두 로그로 남기도록 설정
            new SimpleLoggerAdvisor(Ordered.LOWEST_PRECEDENCE - 1)
        )
        .build();
  }

  // ##### 벡터 저장소의 데이터를 모두 삭제하는 메소드 #####
  // VectorStore의 데이터 삭제 메소드는 필터를 적용한 형태만 가능하기 때문에 
  // 전체 삭제를 위해서 직접 SQL을 실행해서 데이터를 삭제하는 방식으로 구현
  public void clearVectorStore() {
    jdbcTemplate.update("TRUNCATE TABLE vector_store");
  }

  // ##### PDF 파일을 ETL 처리하는 메소드 #####
  // String source :  원본 문서 데이터
  public void ragEtl(MultipartFile attach, String source, int chunkSize, int minChunkSizeChars) throws IOException {
    // 추출하기
    Resource resource = new ByteArrayResource(attach.getBytes());
    DocumentReader reader = new PagePdfDocumentReader(resource);
    List<Document> documents = reader.read();

    // 메타데이터 추가
    for (Document doc : documents) {
      doc.getMetadata().put("source", source);
    }

    // 변환하기
    DocumentTransformer transformer = new TokenTextSplitter(
        chunkSize, minChunkSizeChars, 5, 10000, true,List.of('\n', ' ', '.', ','));
    List<Document> transformedDocuments = transformer.apply(documents);

    // 적재하기
    vectorStore.add(transformedDocuments);
  }

  // ##### LLM과 대화하는 메소드 #####
  // source :  metadata  filter  조건
  public String ragChat(String question, double score, String source) {
    // 벡터 저장소 검색 조건 생성
    SearchRequest.Builder searchRequestBuilder = SearchRequest.builder()
        .similarityThreshold(score)
        .topK(3);
    if (StringUtils.hasText(source)) {
      searchRequestBuilder.filterExpression("source == '%s'".formatted(source));
    }
    SearchRequest searchRequest = searchRequestBuilder.build();


    // QuestionAnswerAdvisor 생성
    QuestionAnswerAdvisor questionAnswerAdvisor = QuestionAnswerAdvisor.builder(vectorStore)
        .searchRequest(searchRequest)
        .build();

    // 프롬프트를 LLM으로 전송하고 응답을 받는 코드
    String answer = this.chatClient.prompt()
        .user(question)
        .advisors(questionAnswerAdvisor)
        .call()
        .content();
    return answer;
  }
}
