/**
 * Mock Database - For testing without Azure SQL
 */

export interface MockUser {
  UserID: number;
  Email: string;
  Username: string;
  PasswordHash: string;
  IsActive: boolean;
  DisplayName: string;
  Tier: string;
  TotalXP: number;
  CurrentLevel: number;
}

// In-memory user storage
const mockUsers: MockUser[] = [];
let nextUserId = 1;

export const mockDatabase = {
  /**
   * Create a new user
   */
  createUser: async (data: {
    Email: string;
    PasswordHash: string;
    Username: string;
    FirstName?: string;
    LastName?: string;
    DisplayName?: string;
  }): Promise<{ UserID: number }> => {
    const userId = nextUserId++;
    
    mockUsers.push({
      UserID: userId,
      Email: data.Email,
      Username: data.Username,
      PasswordHash: data.PasswordHash,
      IsActive: true,
      DisplayName: data.DisplayName || data.Username,
      Tier: 'Bronze',
      TotalXP: 0,
      CurrentLevel: 1,
    });

    console.log(`✅ Mock DB: Created user ${data.Username} (ID: ${userId})`);
    
    return { UserID: userId };
  },

  /**
   * Find user by email or username
   */
  findUser: async (emailOrUsername: string): Promise<MockUser | null> => {
    const user = mockUsers.find(
      (u) => u.Email === emailOrUsername || u.Username === emailOrUsername
    );
    
    return user || null;
  },

  /**
   * Check if user exists
   */
  userExists: async (email: string, username: string): Promise<boolean> => {
    return mockUsers.some(
      (u) => u.Email === email || u.Username === username
    );
  },

  /**
   * Get all users (for debugging)
   */
  getAllUsers: () => mockUsers,
};

export default mockDatabase;



