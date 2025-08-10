<?php

namespace App\Http\Controllers\Auth;

use App\Models\User;
use Inertia\Inertia;
use Illuminate\Http\Request;
use App\Http\Controllers\Controller;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Hash;


class RegisterController extends Controller
{
    public function create()
    {
        return Inertia::render('Auth/Register');
    }

    public function store(Request $request)
    {
        $credentials = $request->validate([
            'username' => ['required', 'string', 'max:255'],
            'email' => ['required', 'email', 'max:255', 'unique:users'],
            'password' => ['required', 'string', 'confirmed'],
        ]);

        if (User::where('email', $request->email)->exists()) {
            return back()->withErrors([
                'email' => 'Registration failed. Please try again.',
            ]);
        }


        $user = User::create([
            'name' => $credentials['username'],
            'email' => $credentials['email'],
            'role' => 'user',
            'password' => Hash::make($credentials['password']),
        ]);

        Auth::login($user);

        return redirect()->intended(route('dashboard'));
    }
}
