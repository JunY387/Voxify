import unittest
import requests

class TestLoginAPI(unittest.TestCase):
    BASE_URL = "http://voxify_api:5000/api/v1/auth/login"
    HEADERS = {"Content-Type": "application/json"}

    # Helper method to send a POST request
    def post_login(self, payload):
        response = requests.post(self.BASE_URL, json=payload, headers=self.HEADERS)
        return response
    
    # TODO: Uncomment and implement this test after registering a user
    # def test_login_successful(self):
    #     # You need to first register a user before logging in!
    #     """Test case for successful login."""
    #     payload = {
    #         "email": "duplicate1@example.com",
    #         "password": "SecurePassword123!"
    #     }
    #     response = self.post_login(payload)
    #     self.assertEqual(response.status_code, 200)
    #     data = response.json()
    #     self.assertIn("access_token", data)
    #     self.assertIn("refresh_token", data)
    #     self.assertIn("user", data)
    #     self.assertEqual(data["user"]["email"], payload["email"])

    def test_login_invalid_email(self):
        """Test case for login with an invalid email."""
        payload = {
            "email": "invaliduser@example.com",
            "password": "ValidPassword123!"
        }
        response = self.post_login(payload)
        self.assertEqual(response.status_code, 401)
        data = response.json()
        self.assertEqual(data["error"]["code"], "INVALID_CREDENTIALS")

    def test_login_invalid_password(self):
        """Test case for login with an invalid password."""
        payload = {
            "email": "validuser@example.com",
            "password": "WrongPassword123!"
        }
        response = self.post_login(payload)
        self.assertEqual(response.status_code, 401)
        data = response.json()
        self.assertEqual(data["error"]["code"], "INVALID_CREDENTIALS")

    def test_login_missing_email(self):
        """Test case for missing email in the request payload."""
        payload = {
            "password": "ValidPassword123!"
        }
        response = self.post_login(payload)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["error"]["code"], "MISSING_FIELDS")

    def test_login_missing_password(self):
        """Test case for missing password in the request payload."""
        payload = {
            "email": "validuser@example.com"
        }
        response = self.post_login(payload)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["error"]["code"], "MISSING_FIELDS")

    def test_login_empty_payload(self):
        """Test case for an empty request payload."""
        payload = {}
        response = self.post_login(payload)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data['error']['code'], "MISSING_BODY")

    def test_login_sql_injection(self):
        """Test case for possible SQL injection."""
        payload = {
            "email": "' OR '1'='1",
            "password": "' OR '1'='1"
        }
        response = self.post_login(payload)
        self.assertEqual(response.status_code, 401)
        data = response.json()
        self.assertEqual(data['error']['code'], "INVALID_CREDENTIALS")

if __name__ == "__main__":
    unittest.main()
