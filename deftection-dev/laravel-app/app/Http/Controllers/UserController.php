<?php

namespace App\Http\Controllers;

use App\Models\User;
use Inertia\Inertia;
use Illuminate\Http\Request;
use App\Services\UserQueryService;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Hash;

class UserController extends Controller
{
    public function __construct(
        protected UserQueryService $userService
    ) {}


    /**
     * Display a listing of users.
     */
    public function index(Request $request)
    {
        try {


            $filters = $this->userService->filters($request);
            $users = $this->userService->getAllUsers($filters);
            $checksum = $this->userService->getChecksumAllUsers();

            // Prepare filter options for frontend
            $filterOptions = [
                'roles' => $this->userService->getRolesWithCounts(),
                'perPageOptions' => [5, 10, 25, 50, 100],
            ];

            return Inertia::render('Database/User/Index', [
                'users' => $users->items(),
                'filters' => [
                    'current' => $filters,
                    'options' => $filterOptions,
                ],
                'meta' => [
                    'total' => $users->total(),
                    'from' => $users->firstItem(),
                    'to' => $users->lastItem(),
                    'current_page' => $users->currentPage(),
                    'last_page' => $users->lastPage(),
                    'per_page' => $users->perPage(),
                ],
                'initialChecksum' => $checksum,
            ]);
        } catch (\Exception $e) {
            Log::error('Error loading users index page', [
                'error' => $e->getMessage(),
                'user_id' => auth()->id(),
                'filters' => $request->all(),
            ]);

            return redirect()
                ->back()
                ->withErrors(['error' => 'Failed to load users. Please try again.']);
        }
    }

    /**
     * Get users data for API (used by polling).
     */
    public function indexApi(Request $request)
    {
        try {


            $filters = $this->userService->filters($request);
            $users = $this->userService->getAllUsers($filters);
            $checksum = $this->userService->getChecksumAllUsers();

            // Prepare filter options for frontend
            $filterOptions = [
                'roles' => $this->userService->getRolesWithCounts(),
                'perPageOptions' => [5, 10, 25, 50, 100],
            ];

            return response()->json([
                'success' => true,
                'users' => $users->items(),
                'filters' => [
                    'current' => $filters,
                    'options' => $filterOptions,
                ],
                'meta' => [
                    'total' => $users->total(),
                    'from' => $users->firstItem(),
                    'to' => $users->lastItem(),
                    'current_page' => $users->currentPage(),
                    'last_page' => $users->lastPage(),
                    'per_page' => $users->perPage(),
                ],
                'checksum' => $checksum,
            ]);
        } catch (\Exception $e) {
            Log::error('Error loading users API data', [
                'error' => $e->getMessage(),
                'user_id' => auth()->id(),
                'filters' => $request->all(),
            ]);

            return response()->json([
                'success' => false,
                'message' => 'Failed to load users data.',
                'error' => 'An error occurred while fetching users data.',
            ], 500);
        }
    }

    /**
     * Force refresh users data (clears any cache if you have it).
     */
    public function indexRefresh(Request $request)
    {
        try {


            $filters = $this->userService->filters($request);
            $users = $this->userService->getAllUsers($filters);
            $checksum = $this->userService->getChecksumAllUsers();

            // Prepare filter options for frontend
            $filterOptions = [
                'roles' => $this->userService->getRolesWithCounts(),
                'perPageOptions' => [5, 10, 25, 50, 100],
            ];

            return response()->json([
                'success' => true,
                'users' => $users,
                'filters' => [
                    'current' => $filters,
                    'options' => $filterOptions,
                ],
                'meta' => [
                    'total' => $users->total(),
                    'from' => $users->firstItem(),
                    'to' => $users->lastItem(),
                    'current_page' => $users->currentPage(),
                    'last_page' => $users->lastPage(),
                    'per_page' => $users->perPage(),
                ],
                'checksum' => $checksum,
            ]);
        } catch (\Exception $e) {
            Log::error('Error force refreshing users data', [
                'error' => $e->getMessage(),
                'user_id' => auth()->id(),
                'filters' => $request->all(),
            ]);

            return response()->json([
                'success' => false,
                'message' => 'Failed to refresh users data.',
                'error' => 'An error occurred while refreshing users data.',
            ], 500);
        }
    }

    /**
     * Check for users data updates (used by polling).
     */
    public function indexCheck(Request $request)
    {
        try {
            $request->validate([
                'checksum' => 'nullable|string',
            ]);


            $currentChecksum = $this->userService->getChecksumAllUsers();
            $providedChecksum = $request->get('checksum', '');

            $hasChanges = $currentChecksum !== $providedChecksum;

            return response()->json([
                'has_changes' => $hasChanges,
                'checksum' => $currentChecksum,
            ]);
        } catch (\Exception $e) {
            Log::error('Error checking users data updates', [
                'error' => $e->getMessage(),
                'user_id' => auth()->id(),
                'provided_checksum' => $request->get('checksum'),
            ]);

            return response()->json([
                'success' => false,
                'message' => 'Failed to check for users data updates.',
                'error' => 'An error occurred while checking for updates.',
            ], 500);
        }
    }

    /**
     * Store a newly created user in storage.
     */
    public function store(Request $request)
    {
        $request->validate([
            'name' => 'required|string|min:2|max:255',
            'email' => 'required|string|email|max:255|unique:users,email',
            'role' => 'required|string|in:admin,user',
            'password' => 'required|string|min:8|max:255',
            'password_confirmation' => 'required|string|same:password',
        ]);

        $user = User::create([
            'name' => trim($request->name),
            'email' => strtolower(trim($request->email)),
            'role' => $request->role,
            'password' => Hash::make($request->password),
        ]);

        Log::info('User created successfully', [
            'user_id' => $user->id,
            'email' => $user->email,
            'role' => $user->role,
            'created_by' => auth()->id(),
        ]);

        return redirect()->back()->with('success', 'User created successfully!');
    }

    /**
     * Update the specified user in storage.
     */
    public function update(Request $request, User $user)
    {
        $rules = [
            'name' => 'required|string|min:2|max:255',
            'email' => 'required|string|email|max:255|unique:users,email,' . $user->id,
            'role' => 'required|string|in:admin,user',
        ];

        if ($request->filled('password')) {
            $rules['password'] = 'required|string|min:8|max:255';
            $rules['password_confirmation'] = 'required|string|same:password';
        }

        $request->validate($rules);

        // Prevent last admin from being demoted
        if ($user->role === 'admin' && $request->role !== 'admin') {
            $adminCount = User::where('role', 'admin')->count();
            if ($adminCount <= 1) {
                return redirect()
                    ->back()
                    ->withErrors(['role' => 'Cannot demote the last admin user.']);
            }
        }

        $originalData = $user->only(['name', 'email', 'role']);

        $updateData = [
            'name' => trim($request->name),
            'email' => strtolower(trim($request->email)),
            'role' => $request->role,
        ];

        if ($request->filled('password')) {
            $updateData['password'] = Hash::make($request->password);
        }

        $user->update($updateData);

        Log::info('User updated successfully', [
            'user_id' => $user->id,
            'original_data' => $originalData,
            'updated_data' => $user->only(['name', 'email', 'role']),
            'password_changed' => $request->filled('password'),
            'updated_by' => auth()->id(),
        ]);

        return redirect()->back()->with('success', 'User updated successfully!');
    }

    /**
     * Remove the specified user from storage.
     */
    public function destroy(User $user)
    {
        // Prevent deletion of the last admin
        if ($user->role === 'admin') {
            $adminCount = User::where('role', 'admin')->count();
            if ($adminCount <= 1) {
                return redirect()
                    ->back()
                    ->withErrors(['error' => 'Cannot delete the last admin user.']);
            }
        }

        $userData = $user->only(['id', 'name', 'email', 'role']);

        $user->delete();

        Log::info('User deleted successfully', [
            'deleted_user' => $userData,
            'deleted_by' => auth()->id(),
        ]);

        return redirect()->back()->with('success', 'User deleted successfully!');
    }

    /**
     * Get user data for API (used by edit modal).
     */
    public function showApi(User $user)
    {
        try {
            return response()->json([
                'success' => true,
                'data' => [
                    'id' => $user->id,
                    'name' => $user->name,
                    'email' => $user->email,
                    'role' => $user->role,
                    'created_at' => $user->created_at->toISOString(),
                    'updated_at' => $user->updated_at->toISOString(),
                    'email_verified_at' => $user->email_verified_at?->toISOString(),
                ],
            ]);
        } catch (\Exception $e) {
            Log::error('Error showing user data: ' . $e->getMessage(), [
                'product_id' => $user->id,
                'user_id' => auth()->id(),
                'trace' => $e->getTraceAsString()
            ]);

            return response()->json([
                'success' => false,
                'error' => 'Error fetching user data. Please try again later.',
            ], 500);
        }
    }
}
