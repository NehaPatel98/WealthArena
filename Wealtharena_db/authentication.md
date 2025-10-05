# WealthArena Authentication Strategy

## Technology Stack
- **JWT (JSON Web Tokens)** for stateless authentication
- **bcrypt** for password hashing
- **OAuth2** ready for future social login integration

## Security Framework

### 1. Password Security
- Passwords hashed using bcrypt (salt rounds: 12)
- Minimum password length: 8 characters
- Password complexity requirements enforced

### 2. JToken Configuration
- Algorithm: HS256
- Expiration: 24 hours
- Secret key: Environment variable (JWT_SECRET_KEY)

### 3. API Security Measures
- Rate limiting: 100 requests/minute per user
- CORS configured for web app domains
- HTTPS enforcement in production
- Input validation on all endpoints

### 4. Database Security
- Row Level Security (RLS) enabled on user data
- Password hashes stored separately from user profiles
- Audit logs for authentication events

### 5. Environment Variables