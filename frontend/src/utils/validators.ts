/**
 * Validates Google Drive folder URL
 * 
 * @param url - URL to validate
 * @returns true if URL is a valid Google Drive folder URL
 */
export const isValidGoogleDriveUrl = (url: string): boolean => {
  if (!url || typeof url !== 'string') {
    return false;
  }

  // Support multiple Google Drive URL formats
  const patterns = [
    /^https:\/\/drive\.google\.com\/drive\/folders\/[a-zA-Z0-9_-]+/,
    /^https:\/\/drive\.google\.com\/drive\/u\/\d+\/folders\/[a-zA-Z0-9_-]+/,
    /^https:\/\/drive\.google\.com\/open\?id=[a-zA-Z0-9_-]+/,
  ];

  return patterns.some((pattern) => pattern.test(url.trim()));
};

/**
 * Extracts folder ID from Google Drive URL
 * 
 * @param url - Google Drive URL
 * @returns Folder ID or null if not found
 */
export const extractFolderId = (url: string): string | null => {
  if (!url || typeof url !== 'string') {
    return null;
  }

  // Try different URL formats
  const patterns = [
    /\/folders\/([a-zA-Z0-9_-]+)/,
    /[?&]id=([a-zA-Z0-9_-]+)/,
  ];

  for (const pattern of patterns) {
    const match = url.match(pattern);
    if (match && match[1]) {
      return match[1];
    }
  }

  return null;
};

export const validateEmail = (email: string): boolean => {
  const pattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return pattern.test(email);
};

