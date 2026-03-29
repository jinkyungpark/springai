package com.example.rag.service;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.document.Document;
import org.springframework.ai.document.DocumentReader;
import org.springframework.ai.document.DocumentTransformer;
import org.springframework.ai.model.transformer.KeywordMetadataEnricher;
import org.springframework.ai.reader.JsonMetadataGenerator;
import org.springframework.ai.reader.JsonReader;
import org.springframework.ai.reader.TextReader;
import org.springframework.ai.reader.jsoup.JsoupDocumentReader;
import org.springframework.ai.reader.jsoup.config.JsoupDocumentReaderConfig;
import org.springframework.ai.reader.pdf.PagePdfDocumentReader;
import org.springframework.ai.reader.tika.TikaDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import lombok.extern.slf4j.Slf4j;

@Service
@Slf4j
public class ETLService {
  // ##### 필드 #####
  private ChatModel chatModel;
  private VectorStore vectorStore;

  // ##### 생성자 #####
  public ETLService(ChatModel chatModel, VectorStore vectorStore) {
    this.chatModel = chatModel;
    this.vectorStore = vectorStore;
  }

  // ##### 업로드된 파일을 가지고 ETL 과정을 처리하는 메소드 #####
  public String etlFromFile(String title, String author,
      MultipartFile attach) throws IOException {

    // 추출하기
    List<Document> documents = extractFromFile(attach);
    if (documents == null) {
      return ".txt, .pdf, .doc, .docx 파일 중에 하나를 올려주세요.";
    }
    log.info("추출된 Document 수: {} 개", documents.size());

    // 메타데이터에 공통 정보 추가하기
    for (Document doc : documents) {
      Map<String, Object> metadata = doc.getMetadata();
      metadata.putAll(Map.of(
          "title", title,
          "author", author,
          "source", attach.getOriginalFilename()));
    }

    // 변환하기
    documents = transform(documents);
    log.info("변환된 Document 수: {} 개", documents.size());

    // 적재하기
    vectorStore.add(documents);

    return "올린 문서를 추출-변환-적재 완료 했습니다.";
  }

  // ##### 업로드된 파일로부터 텍스트를 추출하는 메소드 #####
  private List<Document> extractFromFile(MultipartFile attach) throws IOException {
    // 바이트 배열을 Resource로 생성
    Resource resource = new ByteArrayResource(attach.getBytes());

    List<Document> documents = null;

    if(attach.getContentType() != null) {
      if (attach.getContentType().equals("text/plain")) {
        // Text(.txt) 파일일 경우
        DocumentReader reader = new TextReader(resource);
        documents = reader.read();
      } else if (attach.getContentType().equals("application/pdf")) {
        // PDF(.pdf) 파일일 경우
        DocumentReader reader = new PagePdfDocumentReader(resource);
        documents = reader.read();
      } else if (attach.getContentType().contains("wordprocessingml")) {
        // Word(.doc, .docx) 파일일 경우
        DocumentReader reader = new TikaDocumentReader(resource);
        documents = reader.read();
      }      
    }
    

    return documents;
  }

  // ##### 작은 크기로 분할하고 키워드 메타데이터를 추가하는 메소드 #####
  private List<Document> transform(List<Document> documents) {
    List<Document> transformedDocuments = null;

    // 작게 분할하기(기본값으로 분할)
    TokenTextSplitter tokenTextSplitter = new TokenTextSplitter();
    transformedDocuments = tokenTextSplitter.apply(documents);

    // 메타데이터에 키워드 추가(시간이 많이 걸리는 작업 추천작업 아님)
    // 자른 후에 메타데이터 추가
    // chatModel 을 사용해서 각 Document의 텍스트에서 키워드를 추출해서 메타데이터에 추가하는 작업
    // 이 작업은 단순 키워드 검색으로 가능한 상황이라 유사도 검색에서 의미가 없는 작업
    // KeywordMetadataEnricher keywordMetadataEnricher = 
    //     new KeywordMetadataEnricher(chatModel, 5);
    // transformedDocuments = keywordMetadataEnricher.apply(transformedDocuments);

    return transformedDocuments;
  }

  // ##### HTML의 ETL 과정을 처리하는 메소드 #####
  public String etlFromHtml(String title, String author, String url) throws Exception {
    // URL로부터 Resource 얻기
    Resource resource = new UrlResource(url);

    // E: 추출하기
    JsoupDocumentReader reader = new JsoupDocumentReader(
        resource,
        JsoupDocumentReaderConfig.builder()  
            .charset("UTF-8")
            //쉼표로 구분해서 여러개의 셀렉터 추가 가능
            .selector("#content")
            .metadataTag("author")
            .additionalMetadata(Map.of(
                //"title", title,
                //"author", author,
                "url", url))
            .build());
    List<Document> documents = reader.read();
    log.info("추출된 Document 수: {} 개", documents.size());

    // T: 변환하기
    DocumentTransformer transformer = new TokenTextSplitter();
    List<Document> transformedDocuments = transformer.apply(documents);
    log.info("변환된 Document 수: {} 개", transformedDocuments.size());

    // L: 적재하기
    vectorStore.add(transformedDocuments);

    return "HTML에서 추출-변환-적재 완료 했습니다.";
  }

  // ##### JSON의 ETL 과정을 처리하는 메소드 #####
  public String etlFromJson(String url) throws Exception {
    // URL로부터 Resource 얻기
    Resource resource = new UrlResource(url);

    // E: 추출하기
    JsonReader reader = new JsonReader(
        resource,        
        new JsonMetadataGenerator() { //  메타데이터 저장할 부분 
          @Override
          public Map<String, Object> generate(Map<String, Object> jsonMap) {
            return Map.of(
                "title", jsonMap.get("title"),
                "author", jsonMap.get("author"),
                "url", "http://localhost:8080/document/constitution(19880225).json");
          }
        },

        /*jsonMap -> Map.of(
            "title", jsonMap.get("title"),
            "author", jsonMap.get("author"),
            "url", "http://localhost:8080/document/constitution(19880225).json"
        ),*/

        "date", "content"

    );
    
    List<Document> documents = reader.read();
    log.info("추출된 Document 수: {} 개", documents.size());

    // T: 변환하기
    DocumentTransformer transformer = new TokenTextSplitter();
    List<Document> transformedDocuments = transformer.apply(documents);
    log.info("변환된 Document 수: {} 개", transformedDocuments.size());

    // L: 적재하기
    vectorStore.add(transformedDocuments);

    return "JSON에서 추출-변환-적재 완료 했습니다.";
  }
}
