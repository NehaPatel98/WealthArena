package com.wealtharena.controller;

import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/test")
public class TestController {
    
    @GetMapping("/scheduled")
    public String testScheduled() {
        return "Scheduled data service is working!";
    }
    
    @PostMapping("/trigger")
    public String triggerTest() {
        return "Manual trigger test successful!";
    }
}
