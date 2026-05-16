# Game Entity Relationship Diagram

```mermaid
erDiagram
    BOT {
        string sid PK "folder name"
        string displayName
        string image
        string model
    }
    LOCATION {
        string name PK "folder name"
        string displayName
        string image
    }
    GAME {
        string key PK
    }
    ROLE {
        string name PK "folder name"
        string displayName
        string systemPrompt
        string greeting
        string defaultLocation FK
    }
    CHARACTER {
        string sid PK
        string role FK
        string displayName
        bool isBot "FK to BOT only if true"
    }
    POSITION {
        string sid FK
        string location FK
    }

    LOCATION ||--o{ LOCATION : "connects to"
    GAME ||--o{ ROLE : has
    BOT ||--o{ CHARACTER : sid
    LOCATION ||--o{ ROLE : defaultLocation
    ROLE ||--o{ CHARACTER : role
    CHARACTER ||--o| POSITION : sid
    LOCATION ||--o{ POSITION : location
```
