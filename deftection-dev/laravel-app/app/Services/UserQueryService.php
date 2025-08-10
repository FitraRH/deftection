<?php

namespace App\Services;

use App\Models\User;
use Illuminate\Http\Request;
use Illuminate\Support\Carbon;
use Illuminate\Contracts\Pagination\LengthAwarePaginator;

class UserQueryService
{
    public function getAllUsers(array $filters): LengthAwarePaginator
    {
        return User::query()
            ->when(!empty($filters['search']), function ($query) use ($filters) {
                $searchTerm = trim($filters['search']);
                $query->where(function ($subQuery) use ($searchTerm) {
                    $subQuery->where('name', 'like', '%' . $searchTerm . '%')
                        ->orWhere('email', 'like', '%' . $searchTerm . '%')
                        ->orWhere('id', 'like', '%' . $searchTerm . '%');
                });
            })
            ->when(!empty($filters['roles']), function ($query) use ($filters) {
                $query->whereIn('role', $filters['roles']);
            })
            ->when(!empty($filters['dateFrom']), function ($query) use ($filters) {
                $query->whereDate('created_at', '>=', $filters['dateFrom']);
            })
            ->when(!empty($filters['dateTo']), function ($query) use ($filters) {
                $query->whereDate('created_at', '<=', $filters['dateTo']);
            })
            ->orderBy($filters['sortBy'], $filters['sortDir'])
            ->paginate($filters['perPage'], ['*'], 'page', $filters['page'])
            ->withQueryString()
            ->through(fn($user) => [
                'id' => $user->id,
                'name' => $user->name,
                'email' => $user->email,
                'role' => $user->role,
                'created_at' => $user->created_at->format('Y-m-d H:i:s'),
                'updated_at' => $user->updated_at->format('Y-m-d H:i:s'),
                'email_verified_at' => $user->email_verified_at?->format('Y-m-d H:i:s'),
            ]);
    }

    public function getChecksumAllUsers(): string
    {
        $userData = User::selectRaw('
            COUNT(*) as total_users,
            MAX(updated_at) as latest_update,
            SUM(CASE WHEN role = "admin" THEN 1 ELSE 0 END) as admin_count,
            SUM(CASE WHEN role = "user" THEN 1 ELSE 0 END) as user_count
        ')->first();

        $checksumData = [
            'total_users' => $userData->total_users ?? 0,
            'admin_count' => $userData->admin_count ?? 0,
            'user_count' => $userData->user_count ?? 0,
            'latest_update' => $userData->latest_update
                ? Carbon::parse($userData->latest_update)->timestamp
                : 0,
        ];

        return md5(json_encode($checksumData));
    }

    /**
     * Get all roles with counts
     */
    public function getRolesWithCounts(): array
    {
        return User::select('role')
            ->selectRaw('COUNT(*) as count')
            ->groupBy('role')
            ->orderBy('count', 'desc')
            ->orderBy('role', 'asc')
            ->get()
            ->map(function ($role) {
                return [
                    'label' => ucfirst($role->role),
                    'value' => $role->role,
                    'count' => $role->count,
                ];
            })
            ->toArray();
    }

    /**
     * Get and validate filters from request
     */
    public function filters(Request $request): array
    {
        // Get available role values dynamically
        $availableRoles = User::distinct()->pluck('role')->toArray();

        $validated = $request->validate([
            'search' => 'nullable|string|max:255',
            'sort_by' => 'nullable|string|in:id,name,email,role,created_at,updated_at',
            'sort_dir' => 'nullable|string|in:asc,desc',
            'per_page' => 'nullable|integer|min:1|max:100',
            'page' => 'nullable|integer|min:1',
            'roles' => 'nullable|array',
            'roles.*' => 'string|in:' . implode(',', $availableRoles),
            'date_from' => 'nullable|date',
            'date_to' => 'nullable|date|after_or_equal:date_from',
        ]);

        return [
            'search' => $validated['search'] ?? '',
            'sortBy' => $validated['sort_by'] ?? 'created_at',
            'sortDir' => $validated['sort_dir'] ?? 'desc',
            'perPage' => $validated['per_page'] ?? 10,
            'page' => $validated['page'] ?? 1,
            'roles' => $validated['roles'] ?? [],
            'dateFrom' => $validated['date_from'] ?? null,
            'dateTo' => $validated['date_to'] ?? null,
        ];
    }
}
