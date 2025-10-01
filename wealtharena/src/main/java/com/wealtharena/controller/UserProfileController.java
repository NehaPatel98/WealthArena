package com.wealtharena.controller;

import com.wealtharena.model.User;
import com.wealtharena.repository.UserRepository;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.security.Principal;

@RestController
@RequestMapping("/api/users")
public class UserProfileController {

    private final UserRepository userRepository;

    public UserProfileController(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public record UpdateProfileRequest(String fullName) {}

    @GetMapping("/profile")
    public ResponseEntity<?> getProfile(@RequestParam String email) {
        return userRepository.findByEmail(email)
                .map(u -> ResponseEntity.ok(new Profile(u.getId(), u.getEmail(), u.getFullName())))
                .orElse(ResponseEntity.notFound().build());
    }

    @PutMapping("/profile")
    public ResponseEntity<?> updateProfile(@RequestParam String email, @RequestBody UpdateProfileRequest req) {
        return userRepository.findByEmail(email)
                .map(u -> {
                    u.setFullName(req.fullName());
                    userRepository.save(u);
                    return ResponseEntity.ok(new Profile(u.getId(), u.getEmail(), u.getFullName()));
                })
                .orElse(ResponseEntity.notFound().build());
    }

    public record Profile(Long id, String email, String fullName) {}
}


