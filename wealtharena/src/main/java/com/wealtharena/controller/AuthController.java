package com.wealtharena.controller;

import com.wealtharena.model.User;
import com.wealtharena.repository.UserRepository;
import com.wealtharena.service.JwtService;
import org.springframework.http.ResponseEntity;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/auth")
public class AuthController {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtService jwtService;

    public AuthController(UserRepository userRepository,
                          PasswordEncoder passwordEncoder,
                          JwtService jwtService) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
        this.jwtService = jwtService;
    }

    public record RegisterRequest(String email, String password, String fullName) {}
    public record LoginRequest(String email, String password) {}

    @PostMapping("/register")
    public ResponseEntity<?> register(@RequestBody RegisterRequest req) {
        if (req.email() == null || req.email().isBlank() || req.password() == null || req.password().length() < 6) {
            return ResponseEntity.badRequest().body("Invalid email or password too short");
        }
        if (userRepository.existsByEmail(req.email())) {
            return ResponseEntity.status(409).body("Email already registered");
        }
        User u = new User();
        u.setEmail(req.email());
        u.setFullName(req.fullName());
        u.setPasswordHash(passwordEncoder.encode(req.password()));
        userRepository.save(u);
        Map<String, Object> claims = new HashMap<>();
        claims.put("uid", u.getId());
        String token = jwtService.issueToken(u.getEmail(), claims);
        return ResponseEntity.ok(Map.of("token", token));
    }

    @PostMapping("/login")
    public ResponseEntity<?> login(@RequestBody LoginRequest req) {
        return userRepository.findByEmail(req.email())
                .filter(u -> passwordEncoder.matches(req.password(), u.getPasswordHash()))
                .<ResponseEntity<?>>map(u -> {
                    Map<String, Object> claims = new HashMap<>();
                    claims.put("uid", u.getId());
                    String token = jwtService.issueToken(u.getEmail(), claims);
                    return ResponseEntity.ok(Map.of("token", token));
                })
                .orElse(ResponseEntity.status(401).body("Invalid credentials"));
    }
}


