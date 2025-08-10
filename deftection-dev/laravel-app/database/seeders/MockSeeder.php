<?php

namespace Database\Seeders;

use App\Models\Scan;
use App\Models\ScanDefect;
use App\Models\ScanThreat;
use App\Models\User;
use Illuminate\Database\Seeder;
use Illuminate\Database\Console\Seeds\WithoutModelEvents;

class MockSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {


        User::factory(25)->create()->each(function ($user, $index) {
            if ($index == 0) {
                $user->update([
                    'name' => 'admin name ' . $index,
                    'role' => 'admin',
                    'email' => 'admin@example.com',
                    'password' => bcrypt('123456'),
                ]);
            }
            if ($index == 1) {
                $user->update([
                    'name' => 'user name ' . $index,
                    'role' => 'user',
                    'email' => 'user@example.com',
                    'password' => bcrypt('123456'),
                ]);
            }
            if ($index == 2) {
                $user->update([
                    'name' => 'satria user',
                    'role' => 'guest',
                    'email' => 'satriauser@example.com',
                    'password' => bcrypt('123456'),
                ]);
            }
            if ($index == 3) {
                $user->update([
                    'name' => 'satria admin',
                    'role' => 'admin',
                    'email' => 'satriaadmin@example.com',
                    'password' => bcrypt('123456'),
                ]);
            }

            Scan::factory(25)->create([
                'user_id' => $user->id,
            ])->each(function ($scan) {

                if ($scan->is_defect) {
                    $rand1 = rand(1, 5);

                    ScanDefect::factory($rand1)->create([
                        'scan_id' => $scan->id,
                    ]);
                }

                $rand2 = rand(0, 2);
                if ($rand2 == 0) {
                    ScanThreat::factory()->create([
                        'scan_id' => $scan->id,
                        'status' => 'clean',
                        'risk_level' => 'none',
                        'flags' => null,
                        'details' => null,
                        'possible_attack' => null,
                    ]);
                } elseif ($rand2 == 1) {
                    ScanThreat::factory()->create([
                        'scan_id' => $scan->id,
                    ]);
                }
            });
        });
    }
}
