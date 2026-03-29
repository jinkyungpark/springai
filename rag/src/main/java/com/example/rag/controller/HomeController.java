package com.example.rag.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class HomeController {
  @GetMapping("/")
  public String home() {
    return "home";
  }

  @GetMapping("/txt-pdf-word-etl")
  public String txtPdfDocxEtl() {
    return "txt-pdf-word-etl";
  }

  @GetMapping("/html-etl")
  public String htmlEtl() {
    return "html-etl";
  }

  @GetMapping("/json-etl")
  public String jsonEtl() {
    return "json-etl";
  }

  @GetMapping("/rag")
  public String rag() {
    return "rag";
  }

  @GetMapping("/compression-query-transformer")
  public String compressionQueryTransformer() {
    return "compression-query-transformer";
  }  

  @GetMapping("/rewrite-query-transformer")
  public String rewriteQueryTransformer() {
    return "rewrite-query-transformer";
  }   

  @GetMapping("/translation-query-transformer")
  public String translationQueryTransformer() {
    return "translation-query-transformer";
  }     

  @GetMapping("/multi-query-expander")
  public String multiQueryExpander() {
    return "multi-query-expander";
  }   
}
