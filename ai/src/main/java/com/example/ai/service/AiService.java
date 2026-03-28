package com.example.ai.service;

import java.util.List;
import java.util.Map;

import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingOptions;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.ai.embedding.EmbeddingResponseMetadata;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import lombok.extern.slf4j.Slf4j;

@Service
@Slf4j
public class AiService {
  @Autowired
  private EmbeddingModel embeddingModel;

  // 벡터 저장소 :  DAO
  // 어떤 벡터 저장소를 사용하던 상관없이
  // implementation 'org.springframework.ai:spring-ai-starter-vector-store-pgvector'
  @Autowired
  private VectorStore vectorStore;


  // 임베딩 + 다른 정보
  // public void textEmbedding(String question) {
  //   // 임베딩하기
  //   EmbeddingResponse response = embeddingModel.embedForResponse(List.of(question));

  //   // 임베딩 모델 정보 얻기
  //   EmbeddingResponseMetadata metadata = response.getMetadata();
  //   log.info("모델명 : {}", metadata.getModel());
  //   log.info("모델 임베딩 차원 : {}", embeddingModel.dimensions());

  //   // 임베딩 결과 얻기
  //   Embedding embedding = response.getResults().get(0);
  //   log.info("벡터 차원: {}", embedding.getOutput().length);
  //   log.info("벡터: {}", embedding.getOutput());
  // } 
  
  // 임베딩 
  // public void textEmbedding(String question) {
  //   float[] vector = embeddingModel.embed(question);
  //   log.info("벡터 차원: {}", vector.length);
  //   log.info("벡터: {}", vector);
  // }


  // application.yml에서 설정한 옵션을 사용하지 않는다면
  public void textEmbedding(String question) {
    EmbeddingOptions options = EmbeddingOptions.builder()
        .model("text-embedding-3-small")  // text-embedding-ada-002(기본)
        .build();
    EmbeddingRequest request = new EmbeddingRequest(List.of(question), options);

    EmbeddingResponse response = embeddingModel.call(request);
    EmbeddingResponseMetadata metadata = response.getMetadata();

    log.info("모델명 : {}", metadata.getModel());
    log.info("벡터 차원: {}", response.getResults().get(0).getOutput().length);
    log.info("벡터: {}", response.getResults().get(0).getOutput());
  }


  // Document
  // VectorStore에 저장할 때 사용하는 객체
  // Langchain의 Document와 유사한 형태로, 텍스트와 메타데이터를 포함하는 객체

  public void addDocument() {
    // Document 목록 생성
    // 각각 레코드로 저장(metadata 는 맘대로 추가 가능)
    // metadata는 벡터 검색 시 필터링 조건으로도 활용 가능
    List<Document> documents = List.of(
        new Document("대통령 선거는 5년마다 있습니다.", Map.of("source", "헌법", "year", 1987)),
        new Document("대통령 임기는 4년입니다.", Map.of("source", "헌법", "year", 1980)),
        new Document("국회의원은 법률안을 심의·의결합니다.", Map.of("source", "헌법", "year", 1987)),
        new Document("자동차를 사용하려면 등록을 해야합니다.", Map.of("source", "자동차관리법")),
        new Document("대통령은 행정부의 수반입니다.", Map.of("source", "헌법", "year", 1987)),
        new Document("국회의원은 4년마다 투표로 뽑습니다.", Map.of("source", "헌법", "year", 1987)),
        new Document("승용차는 정규적인 점검이 필요합니다.", Map.of("source", "자동차관리법")));

    // 벡터 저장소에 저장
    vectorStore.add(documents);
  }

  // 유사도 검색
  // 질문을 받아서 벡터로 변환 -> 벡터 저장소에 유사한 벡터(=문서)가 있는지 검색 -> 유사한 문서 반환
  // 유사도 높은 순서로 상위 4개 반환
  public List<Document> searchDocument1(String question) {
    List<Document> documents = vectorStore.similaritySearch(question);
    return documents;
  }

  // 유사도 검색 + 필터링
  public List<Document> searchDocument2(String question) {
    List<Document> documents = vectorStore.similaritySearch(
        SearchRequest.builder()
            .query(question)
            .topK(1)
            .similarityThreshold(0.4)
            .filterExpression("source == '헌법' && year >= 1987")
            .build());
    return documents;
  }


  // 위 코드를 자바 코드로 사용한다면
  // public List<Document> searchDocument2(String question) {
  //   FilterExpressionBuilder feb = new FilterExpressionBuilder();

  //   List<Document> documents = vectorStore.similaritySearch(
  //       SearchRequest.builder()
  //           .query(question)
  //           .topK(1)
  //           .similarityThreshold(0.4)
  //           .filterExpression(feb
  //               .and(
  //                   feb.eq("source", "헌법"),
  //                   feb.gte("year", 1987))
  //               .build())
  //           .build());
  //   return documents;
  // }

  public void deleteDocument() {
    vectorStore.delete("source == '헌법' && year < 1987");
  }
}
