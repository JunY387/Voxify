import React, { useState, useRef, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import authService from '../services/auth.service';

const Navbar = () => {
  const navigate = useNavigate();
  const user = authService.getCurrentUser();
  const isAuthenticated = authService.isAuthenticated();
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useRef(null);

  // Handle clicking outside to close dropdown
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleLogout = () => {
    authService.logout();
    navigate('/login');
  };

  return (
    <nav className="bg-black border-b border-zinc-800 px-4">
      <div className="h-16 flex items-center justify-between">
        {/* Left section with logo */}
        <div className="flex items-center space-x-2">
          <Link to="/" className="flex items-center">
            <img src="/logo.png" alt="Voxify Logo" className="h-8 w-auto" />
          </Link>
        </div>

        {/* Center section with main navigation */}
        {isAuthenticated && (
          <div className="flex items-center space-x-8">
            <Link to="/dashboard" className="text-white hover:text-gray-300 transition-colors">
              Dashboard
            </Link>
            <Link to="/tasks/list" className="text-white hover:text-gray-300 transition-colors">
              Tasks
            </Link>
            <Link to="/voices" className="text-white hover:text-gray-300 transition-colors">
              Voices
            </Link>
            <Link to="/settings" className="text-white hover:text-gray-300 transition-colors">
              Settings
            </Link>
          </div>
        )}

        {/* Right section with help and profile */}
        {isAuthenticated ? (
          <div className="flex items-center space-x-4">
            <button className="text-white p-2 rounded-full hover:bg-zinc-900 transition-colors">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9 5.25h.008v.008H12v-.008z" />
              </svg>
            </button>
            
            {/* Profile dropdown */}
            <div className="relative" ref={dropdownRef}>
              <button
                onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                className="h-8 w-8 rounded-full bg-orange-300 flex items-center justify-center overflow-hidden hover:ring-2 hover:ring-white/20 transition-all"
              >
                {user?.profile_image ? (
                  <img src={user.profile_image} alt="Profile" className="h-full w-full object-cover" />
                ) : (
                  <span className="text-dark font-medium">{user?.first_name?.[0] || 'U'}</span>
                )}
              </button>

              {/* Dropdown menu */}
              {isDropdownOpen && (
                <div className="absolute right-0 mt-2 w-48 rounded-md shadow-lg bg-zinc-900 ring-1 ring-black ring-opacity-5">
                  <div className="py-1" role="menu" aria-orientation="vertical" aria-labelledby="options-menu">
                    <div className="px-4 py-2 text-sm text-gray-300 border-b border-zinc-800">
                      <p className="font-medium">{user?.first_name || 'User'}</p>
                      <p className="text-gray-400 text-xs mt-0.5">{user?.email}</p>
                    </div>
                    <Link
                      to="/settings"
                      className="block px-4 py-2 text-sm text-gray-300 hover:bg-zinc-800 transition-colors"
                      role="menuitem"
                      onClick={() => setIsDropdownOpen(false)}
                    >
                      Settings
                    </Link>
                    <button
                      onClick={handleLogout}
                      className="block w-full text-left px-4 py-2 text-sm text-red-400 hover:bg-zinc-800 transition-colors"
                      role="menuitem"
                    >
                      Log out
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="flex items-center space-x-4">
            <Link to="/login" className="text-gray-400 hover:text-white transition-colors px-3 py-2">
              Log in
            </Link>
            <Link to="/register" className="border-2 border-white text-white hover:bg-white hover:text-black transition-colors px-4 py-2 rounded">
              Get started
            </Link>
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navbar; 