package com.example.ai.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class HomeController {
  @GetMapping("/")
  public String home() {
    return "home";
  }

  @GetMapping("/text-embedding")
  public String textEmbedding() {
    return "text-embedding";
  }  

  @GetMapping("/add-document")
  public String addDocument() {
    return "add-document";
  }   

  @GetMapping("/search-document-1")
  public String searchDocument1() {
    return "search-document-1";
  }   
  
  @GetMapping("/search-document-2")
  public String searchDocument2() {
    return "search-document-2";
  }  
  
  @GetMapping("/delete-document")
  public String deleteDocument() {
    return "delete-document";
  } 
  
  @GetMapping("/image-embedding")
  public String faceRecognition() {
    return "image-embedding";
  }  
  
}
