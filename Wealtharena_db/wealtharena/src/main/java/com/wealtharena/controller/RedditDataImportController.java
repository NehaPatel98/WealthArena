package com.wealtharena.controller;

import com.wealtharena.service.RedditDataImportService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/reddit")
@CrossOrigin(origins = "*")
public class RedditDataImportController {

    @Autowired
    private RedditDataImportService redditDataImportService;

    @PostMapping("/import")
    public ResponseEntity<String> importRedditData(@RequestParam("file") MultipartFile file) {
        try {
            if (file.isEmpty()) {
                return ResponseEntity.badRequest().body("File is empty");
            }

            if (!file.getOriginalFilename().toLowerCase().endsWith(".csv")) {
                return ResponseEntity.badRequest().body("Please upload a CSV file");
            }

            redditDataImportService.importRedditData(file);
            return ResponseEntity.ok("Reddit data imported successfully!");

        } catch (Exception e) {
            return ResponseEntity.badRequest().body("Import failed: " + e.getMessage());
        }
    }

    @GetMapping("/import/status")
    public ResponseEntity<String> getImportStatus() {
        return ResponseEntity.ok("Reddit data import service is ready. Upload a CSV file to /api/reddit/import");
    }
}
